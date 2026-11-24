//! Host descriptor heaps behind the title's shader-visible heaps (decision
//! 0007, step 3). Translated shaders index the heaps directly at the
//! title's descriptor positions, so each host heap keeps the title heap's
//! order: slot `i` is made from what the title wrote at slot `i`, when it
//! writes it (view creation and descriptor copies) and again when a draw
//! reads it through a table and finds the slot changed.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::{VIEW_TAG, View, view_at};
use crate::host::{Host, HostList, host};
use crate::shadow::Shadow;

/// An aligned upload copy backing a buffer SRV whose offset inside its shadow
/// buffer is not a multiple of its element stride. Kept persistently mapped so
/// each draw or dispatch using the slot can schedule a submission-time refresh
/// (safe to overwrite in place because the host queue waits for each submission
/// to finish before the next one runs).
#[derive(Clone)]
struct UnalignedCopy {
    resource: ID3D12Resource,
    dst: usize,
    src: usize,
    len: usize,
}

/// Submission-time refresh state for a slot: either an aligned copy upload or
/// an aliased view whose exact backing shadow may differ from `buffer_range`.
#[derive(Clone)]
enum SlotRefresh {
    Unaligned(UnalignedCopy),
    Shadow {
        shadow: Arc<Shadow>,
        address: u64,
        len: u64,
    },
}

pub(crate) struct HostHeap {
    pub(crate) heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: u32,
    /// Slots mirroring the title's; `spare` more follow them.
    count: u32,
    /// Dynamic slots past the title's: [`DYNAMIC`] in a view heap where
    /// the host takes them, else [`MIN_DYNAMIC`]; [`SAMPLER_DYNAMIC`] in a
    /// capped sampler heap, none in another.
    spare: u32,
    sampler: bool,
    /// The record (view index and generation) each slot was last made from.
    made: Mutex<Vec<u64>>,
    /// The host resource each slot views, held while the slot does: shaders
    /// index heaps directly, so a slot the title has not rewritten since it
    /// released the resource may still be read, and a freed host resource
    /// read there is a GPU page fault (the console reads stale memory).
    kept: Mutex<Vec<Option<ID3D12Resource>>>,
    /// Slots requiring submission-time title-memory refresh (`UnalignedCopy`
    /// uploads or aliased structured SRV shadows).
    refreshes: Mutex<HashMap<u32, SlotRefresh>>,
    /// Where the search for free dynamic slots starts, and which are taken
    /// (by lists not yet reset).
    next: Mutex<(u32, Vec<bool>)>,
}

/// Host heaps' GPU ranges (start, bytes, bytes per slot, sampler), for
/// device-removed reports: a shader indexing a heap past its end faults.
static GPU_RANGES: Mutex<Vec<(u64, u64, u64, bool)>> = Mutex::new(Vec::new());

/// For device-removed reports: where host GPU address `va` lies against
/// the host descriptor heaps.
pub(crate) fn near_heaps(va: u64) -> String {
    let ranges = GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    for &(start, bytes, step, sampler) in ranges.iter() {
        let kind = if sampler { "sampler" } else { "view" };
        let place = if va < start {
            format!("{:#x} below", start - va)
        } else if va < start + bytes {
            format!("inside, slot {}", (va - start) / step)
        } else {
            format!(
                "{:#x} past its end (slot {})",
                va - start - bytes,
                (va - start) / step
            )
        };
        text.push_str(&format!("; {kind} heap {start:#x}+{bytes:#x}: {place}"));
    }
    text
}

/// The host's limit for shader-visible view heaps on resource binding
/// tiers 1 and 2; tier 3 hardware may take more.
const PORTABLE_VIEWS: u32 = 1_000_000;

/// The largest view heap made: translated shaders send indices at or past
/// it to slot 0, so a larger heap would lose its top slots to them.
const MAX_VIEWS: u32 = victory_shader::emit::VIEW_HEAP_LIMIT;
const _: () = assert!(PORTABLE_VIEWS <= MAX_VIEWS && DYNAMIC < PORTABLE_VIEWS);
const _: () = assert!(SAMPLERS <= victory_shader::emit::SAMPLER_HEAP_LIMIT);

/// Slots past the title's in a view heap, for tables a draw needs made
/// afresh (constant buffers copied per draw). A list holds the slots it
/// took until it is reset (its submissions have run) or released.
/// 2^16 ran out seconds into the world (every world run on an RX 9070 XT,
/// each a device removal shortly after; #59).
const DYNAMIC: u32 = 1 << 18;

/// The dynamic slots of a heap whose host would not take [`DYNAMIC`] more
/// (a capped heap, NVIDIA): the title's slots keep their room first.
const MIN_DYNAMIC: u32 = 1 << 16;

/// The host's limit for shader-visible sampler heaps, on every binding
/// tier.
const SAMPLERS: u32 = D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE;

/// Dynamic slots of a sampler heap capped at [`SAMPLERS`], for tables
/// reaching past the title slots it keeps.
const SAMPLER_DYNAMIC: u32 = 256;

impl HostHeap {
    pub(crate) fn new(sampler: bool, count: u32) -> Option<Self> {
        let host = host()?;
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        let make = |count: u32| -> windows::core::Result<ID3D12DescriptorHeap> {
            // SAFETY: plain heap creation.
            unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            }
        };
        if sampler && count > SAMPLERS {
            // shortcut: samplers past the host's limit are left out; tables
            // reaching them are made afresh in the dynamic slots.
            tracing::warn!(count, "host sampler heap capped at {SAMPLERS}");
            let heap = match make(SAMPLERS) {
                Ok(heap) => heap,
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            };
            return Self::with(
                host,
                heap,
                kind,
                SAMPLERS - SAMPLER_DYNAMIC,
                SAMPLER_DYNAMIC,
            );
        }
        if sampler {
            return match make(count) {
                Ok(heap) => Self::with(host, heap, kind, count, 0),
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    None
                }
            };
        }
        // The title slots that fit under the largest heap made beside
        // `extra` dynamic ones.
        let fitting = |extra: u32| count.min(MAX_VIEWS - extra);
        if fitting(MIN_DYNAMIC) < count {
            // shortcut: as below, descriptors past the limit are left out.
            tracing::warn!(count, "host view heap capped at {MAX_VIEWS}");
        }
        // The larger dynamic range only where the host takes it on top of
        // every title slot; else the smaller one, and the cap after that.
        let large = if fitting(DYNAMIC) == count {
            make(count + DYNAMIC).ok()
        } else {
            None
        };
        let (heap, count, extra) = if let Some(heap) = large {
            (heap, count, DYNAMIC)
        } else {
            let count = fitting(MIN_DYNAMIC);
            match make(count + MIN_DYNAMIC) {
                Ok(heap) => (heap, count, MIN_DYNAMIC),
                Err(e) if count + MIN_DYNAMIC > PORTABLE_VIEWS => {
                    // shortcut: descriptors past the host's limit are left
                    // out; a title that reaches them needs remapping.
                    tracing::warn!(error = %e, count, "host view heap capped at {PORTABLE_VIEWS}");
                    (
                        make(PORTABLE_VIEWS).ok()?,
                        PORTABLE_VIEWS - MIN_DYNAMIC,
                        MIN_DYNAMIC,
                    )
                }
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        };
        Self::with(host, heap, kind, count, extra)
    }

    /// The host heap over `heap`: `count` title slots, then `extra` dynamic
    /// ones, all made null.
    fn with(
        host: &Host,
        heap: ID3D12DescriptorHeap,
        kind: D3D12_DESCRIPTOR_HEAP_TYPE,
        count: u32,
        extra: u32,
    ) -> Option<Self> {
        let sampler = kind == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
        // SAFETY: a live heap.
        let (cpu, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                host.device.GetDescriptorHandleIncrementSize(kind),
            )
        };
        tracing::debug!(count, sampler, "host descriptor heap");
        // SAFETY: a live shader-visible heap.
        let gpu = unsafe { heap.GetGPUDescriptorHandleForHeapStart().ptr };
        GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner()).push((
            gpu,
            u64::from(count + extra) * u64::from(step),
            u64::from(step),
            sampler,
        ));
        let h = HostHeap {
            heap,
            cpu,
            step,
            count,
            spare: extra,
            sampler,
            made: Mutex::new(vec![0; count as usize]),
            kept: Mutex::new(vec![None; (count + extra) as usize]),
            refreshes: Mutex::new(HashMap::new()),
            next: Mutex::new((0, vec![false; extra as usize])),
        };
        // A slot never written is undefined: shaders reading one (a slot
        // the title wrote in a way we do not know, or a table entry no
        // used-resource entry names) read a wild address. Every slot starts
        // null instead.
        let started = std::time::Instant::now();
        if let Err(e) = h.fill_null(kind, count + extra) {
            tracing::warn!(error = %e, "host descriptor heap not made null");
            return None;
        }
        tracing::debug!(
            ms = started.elapsed().as_millis(),
            "host descriptor heap made null"
        );
        Some(h)
    }

    /// Makes the first `total` slots null: one null slot in a CPU-side heap,
    /// doubled by copies to a block, the block copied over the heap.
    fn fill_null(&self, kind: D3D12_DESCRIPTOR_HEAP_TYPE, total: u32) -> windows::core::Result<()> {
        let Some(host) = host() else { return Ok(()) };
        let block_count = total.min(if self.sampler {
            D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE
        } else {
            1 << 16
        });
        // SAFETY: plain heap creation and copies inside live heaps.
        unsafe {
            let block: ID3D12DescriptorHeap =
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: block_count,
                        ..Default::default()
                    })?;
            let start = block.GetCPUDescriptorHandleForHeapStart();
            let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (i * self.step) as usize,
            };
            if self.sampler {
                null_sampler(start);
            } else {
                null_view(start);
            }
            let mut filled = 1;
            while filled < block_count {
                let n = filled.min(block_count - filled);
                host.device
                    .CopyDescriptorsSimple(n, at(filled), start, kind);
                filled += n;
            }
            let mut done = 0;
            while done < total {
                let n = block_count.min(total - done);
                host.device
                    .CopyDescriptorsSimple(n, self.cpu(done), start, kind);
                done += n;
            }
        }
        Ok(())
    }

    /// A plain point sampler at slot `index`, for sampler slots without a
    /// view.
    fn null_sampler(&self, index: u32) {
        null_sampler(self.cpu(index));
    }
}

/// A plain point sampler at `at`.
fn null_sampler(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        ..Default::default()
    };
    // SAFETY: a slot of a sampler heap of ours.
    unsafe { host.device.CreateSampler(&desc, at) };
}

impl HostHeap {
    /// Total dynamic slot capacity of this heap (independent of current free
    /// space), so callers can reject impossible table sizes before scanning.
    pub(crate) fn dynamic_capacity(&self) -> u32 {
        self.spare
    }

    /// `n` consecutive free dynamic slots (view heaps and capped sampler
    /// heaps), taken until [`HostHeap::release`]: the first's index.
    pub(crate) fn dynamic(&self, n: u32) -> Option<u32> {
        if n == 0 || self.spare == 0 {
            return None;
        }
        if n > self.spare {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    n,
                    spare = self.spare,
                    sampler = self.sampler,
                    "a table larger than the dynamic slots is not made"
                );
            });
            return None;
        }
        let spare = self.spare as usize;
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let (next, taken) = &mut *guard;
        let n = n as usize;
        // Search from `next` on, then from the start.
        let starts = (*next as usize..spare).chain(0..*next as usize);
        let mut run = 0;
        let mut found = None;
        let mut last = usize::MAX;
        for at in starts {
            // A run cannot wrap past the end.
            if at != last.wrapping_add(1) {
                run = 0;
            }
            last = at;
            run = if taken[at] { 0 } else { run + 1 };
            if run == n {
                found = Some(at + 1 - n);
                break;
            }
        }
        let Some(first) = found else {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(n, "no free dynamic descriptor slots: a table is not made")
            });
            return None;
        };
        taken[first..first + n].fill(true);
        *next = ((first + n) % spare) as u32;
        Some(self.count + first as u32)
    }

    /// The title slots this heap has host slots for; fewer than the title's
    /// when the host caps its size.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// Gives back `n` dynamic slots from `first` (an index [`dynamic`]
    /// returned).
    ///
    /// [`dynamic`]: HostHeap::dynamic
    pub(crate) fn release(&self, first: u32, n: u32) {
        let Some(at) = first.checked_sub(self.count) else {
            return;
        };
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let end = at.saturating_add(n).min(self.spare) as usize;
        if let Some(slots) = guard.1.get_mut(at as usize..end) {
            slots.fill(false);
        }
    }

    /// Makes slot `index` (a dynamic one) from `view`; a null view if the
    /// host cannot.
    pub(crate) fn make_at(&self, index: u32, view: &View) {
        let kept = make(view, self.cpu(index));
        if kept.is_none() {
            if self.sampler {
                self.null_sampler(index);
            } else {
                null_for(view, self.cpu(index));
            }
        }
        self.keep(index, kept.unwrap_or(Kept::Empty));
    }

    /// Makes slot `index` (a dynamic one) a structured buffer alias with
    /// element width `stride` without mutating the canonical title heap slot.
    pub(crate) fn make_structured_at(&self, index: u32, view: &View, stride: u32) -> bool {
        let Some((address, len, desc)) = buffer_srv_parts(view) else {
            return false;
        };
        let Some(alias) = structured_alias(desc, stride) else {
            return false;
        };
        let Some(kept) = buffer_srv_inner(address, len, alias, self.cpu(index), true) else {
            return false;
        };
        self.keep(index, kept);
        true
    }

    #[cfg(test)]
    pub(crate) fn gpu(&self, index: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: a live shader-visible descriptor heap.
        let base = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart() };
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: base.ptr + u64::from(index) * u64::from(self.step),
        }
    }

    /// Makes slot `index` read nothing: a null view, or a plain sampler.
    pub(crate) fn make_null(&self, index: u32) {
        if self.sampler {
            self.null_sampler(index);
        } else {
            null_view(self.cpu(index));
        }
        self.keep(index, Kept::Empty);
    }

    /// Slot `index` now views `kept` (or nothing of the host's).
    fn keep(&self, index: u32, kept: Kept) {
        let (resource, refresh) = match kept {
            Kept::Empty => (None, None),
            Kept::Resource(r) => (Some(r), None),
            Kept::Unaligned(copy) => (
                Some(copy.resource.clone()),
                Some(SlotRefresh::Unaligned(copy)),
            ),
            Kept::AliasedShadow {
                shadow,
                address,
                len,
            } => (
                Some(shadow.buffer.clone()),
                Some(SlotRefresh::Shadow {
                    shadow,
                    address,
                    len,
                }),
            ),
        };
        let mut slots = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = slots.get_mut(index as usize) {
            // A running submission may still read the old resource through
            // this slot.
            if let Some(old) = std::mem::replace(slot, resource) {
                crate::host::retire(old);
            }
        }
        drop(slots);
        let mut map = self.refreshes.lock().unwrap_or_else(|e| e.into_inner());
        match refresh {
            Some(entry) => {
                map.insert(index, entry);
            }
            None if !map.is_empty() => {
                map.remove(&index);
            }
            None => {}
        }
    }

    /// If slot `index` has a registered refresh entry (`UnalignedCopy` upload
    /// or aliased shadow range), schedules a submission-time refresh on `list`
    /// and returns `true`.
    pub(crate) fn refresh_slot(&self, index: u32, list: &HostList) -> bool {
        let refresh = {
            let map = self.refreshes.lock().unwrap_or_else(|e| e.into_inner());
            if map.is_empty() {
                return false;
            }
            map.get(&index).cloned()
        };
        match refresh {
            Some(SlotRefresh::Unaligned(copy)) => {
                list.defer_fill_upload(
                    copy.resource,
                    copy.dst as *mut u8,
                    copy.src as *const u8,
                    copy.len,
                );
                true
            }
            Some(SlotRefresh::Shadow {
                shadow,
                address,
                len,
            }) => {
                shadow.refresh(list, address, len);
                true
            }
            None => false,
        }
    }

    /// Makes slot `index` (a dynamic one) a constant buffer view of host
    /// memory (the list's own, which it holds).
    pub(crate) fn constant_at(&self, index: u32, gpu: u64, size: u32) {
        self.keep(index, Kept::Empty);
        let Some(host) = host() else { return };
        let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
            BufferLocation: gpu,
            SizeInBytes: size.next_multiple_of(256),
        };
        // SAFETY: a slot of our heap; the memory is the caller's.
        unsafe {
            host.device
                .CreateConstantBufferView(Some(&desc), self.cpu(index))
        };
    }

    /// Makes slot `index` (a dynamic one) a raw buffer view of `len` bytes
    /// of `buffer` from byte `offset` (a multiple of 16): a root SRV with a
    /// size, whose reads past it return zeros instead of reaching whatever
    /// the host placed after the buffer.
    pub(crate) fn raw_at(&self, index: u32, buffer: &ID3D12Resource, offset: u64, len: u64) {
        let Some(host) = host() else { return };
        let elements = (len / 4).clamp(1, u64::from(u32::MAX)) as u32;
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: offset / 4,
                    NumElements: elements,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        // SAFETY: a slot of our heap and a live buffer covering the range.
        unsafe {
            host.device
                .CreateShaderResourceView(buffer, Some(&desc), self.cpu(index))
        };
        self.keep(index, Kept::Resource(buffer.clone()));
    }

    /// Makes slot `index` a raw buffer view that reads zeros.
    pub(crate) fn null_raw_at(&self, index: u32) {
        null_buffer_view(self.cpu(index));
        self.keep(index, Kept::Empty);
    }

    /// Whether this is a sampler heap.
    pub(crate) fn is_sampler(&self) -> bool {
        self.sampler
    }

    /// The shader-visible handle of slot `index`.
    pub(crate) fn gpu_handle(&self, index: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: a live shader-visible heap.
        let start = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart().ptr };
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: start + u64::from(index) * u64::from(self.step),
        }
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + index as usize * self.step as usize,
        }
    }

    /// Makes host slot `index` from the record the title has at `handle`
    /// if it changed since, and returns the view recorded there. Buffer
    /// views are made only for draws (`for_draw`), which also copy their
    /// data: a host copy of a buffer is made on first use.
    pub(crate) fn mirror(&self, index: u32, handle: u64, for_draw: bool) -> Option<View> {
        if index >= self.count {
            // Diagnostic (NVIDIA heap cap): the highest title slot past the
            // host heap seen so far.
            static HIGHEST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if HIGHEST.fetch_max(index, std::sync::atomic::Ordering::Relaxed) < index {
                tracing::warn!(
                    index,
                    count = self.count,
                    for_draw,
                    "title slot past the host heap"
                );
            }
            return None;
        }
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        // Generations start at 1, so a made slot's key is never 0.
        let key = u64::from(generation) << 32 | u64::from(i);
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let view = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten();
        let Some(view) = view else {
            // Not a view of ours (never written, or of a resource the
            // title released): nothing to read.
            if made[index as usize] != 0 {
                made[index as usize] = 0;
                self.make_null(index);
            }
            return None;
        };
        if made[index as usize] != key {
            let later = !for_draw && view.is_buffer();
            made[index as usize] = if later { 0 } else { key };
            let kept = if later {
                None
            } else {
                make(&view, self.cpu(index))
            };
            if kept.is_none() && !self.sampler {
                null_for(&view, self.cpu(index));
            }
            self.keep(index, kept.unwrap_or(Kept::Empty));
        }
        Some(view)
    }

    /// The view host slot `index` was made from, in a few words, with the
    /// title and host addresses of a buffer view (diagnostics). "null" for
    /// a slot never made or left null.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn describe_slot(&self, index: u32) -> Option<String> {
        let key = *self
            .made
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(index as usize)?;
        if key == 0 {
            return Some("null".into());
        }
        let view = view_at(key as u32, (key >> 32) as u32)?;
        let mut text: String = format!("{view:?}").chars().take(120).collect();
        if let Some((address, len)) = buffer_range(&view) {
            let host = crate::shadow::find(address, len.max(1)).map(|s| s.gpu(address));
            text.push_str(&format!(" title {address:#x}+{len:#x} host {host:#x?}"));
        }
        Some(text)
    }
}

/// Logs `what` the first time it happens.
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A view description from its words, with the format as the host has it.
fn srv_desc(words: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>()
            .read_unaligned()
    }
}

fn uav_desc(words: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned()
    }
}

/// A buffer view of title memory at `address` on its host copy: the first
/// element moves by the copy's offset.
/// Bytes per element of a buffer view.
fn element_size(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    }
}

/// The title memory a buffer shader or unordered access view reads: its
/// address and length.
pub(crate) fn buffer_range(view: &View) -> Option<(u64, u64)> {
    let (base, words): (u64, &[u32; 10]) = match view {
        View::Shader(r, Some(desc)) => (
            crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64,
            desc,
        ),
        View::Placed(false, resource, desc) if resource[0] == 1 => (
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            desc[..10].try_into().ok()?,
        ),
        View::Unordered(r, _, Some(desc)) => {
            let base = crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64;
            return unordered_range(base, desc);
        }
        View::Placed(true, resource, desc) if resource[0] == 1 => {
            let base = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            return unordered_range(base, desc[..10].try_into().ok()?);
        }
        _ => return None,
    };
    let desc = srv_desc(words);
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// [`buffer_range`] of an unordered access view of memory at `base`.
fn unordered_range(base: u64, words: &[u32; 10]) -> Option<(u64, u64)> {
    let desc = uav_desc(words);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32)?.0
    };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// Whether a raw buffer view may start at element `first` (4 bytes each):
/// the host wants 16-byte aligned starts, the console only 4, and refuses
/// the call (removing the device) for the rest.
/// shortcut: a view the host refuses reads nothing (a null view); starting
/// it 16 bytes lower would need the shader to add the difference.
fn raw_start_fits(first: u64) -> bool {
    first.is_multiple_of(4)
}

/// The host resource a slot made by [`make`] views, if any.
enum Kept {
    Empty,
    Resource(ID3D12Resource),
    Unaligned(UnalignedCopy),
    AliasedShadow {
        shadow: Arc<Shadow>,
        address: u64,
        len: u64,
    },
}

fn buffer_srv_parts(view: &View) -> Option<(u64, u64, D3D12_SHADER_RESOURCE_VIEW_DESC)> {
    const BUFFER: u32 = 1;
    match view {
        View::Shader(resource, Some(desc)) => {
            let (data, len) = crate::resource::buffer_bytes(*resource as crate::object::This)?;
            Some((data as u64, len, srv_desc(desc)))
        }
        View::Placed(false, resource, desc) if resource[0] == BUFFER => {
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            Some((address, len, srv_desc(desc[..10].try_into().ok()?)))
        }
        _ => None,
    }
}

/// Re-expresses the exact byte start and length of a raw buffer SRV as
/// structured elements of width `stride`, rejecting typed or already-structured
/// views and fractional bounds.
fn structured_alias(
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    stride: u32,
) -> Option<D3D12_SHADER_RESOURCE_VIEW_DESC> {
    if desc.ViewDimension != D3D12_SRV_DIMENSION_BUFFER
        || !(4..=2048).contains(&stride)
        || !stride.is_multiple_of(4)
    {
        return None;
    }
    let old_bytes = element_size(&desc);
    // SAFETY: `ViewDimension == D3D12_SRV_DIMENSION_BUFFER` verified above.
    let old = unsafe { desc.Anonymous.Buffer };
    if old.StructureByteStride != 0
        || old.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 == 0
        || old_bytes == 0
        || old.NumElements == 0
    {
        return None;
    }
    let width = u64::from(stride);
    let start = old.FirstElement.checked_mul(old_bytes)?;
    let size = u64::from(old.NumElements).checked_mul(old_bytes)?;
    if start % width != 0 || size % width != 0 {
        return None;
    }
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.Anonymous.Buffer = D3D12_BUFFER_SRV {
        FirstElement: start / width,
        NumElements: u32::try_from(size / width).ok()?,
        StructureByteStride: stride,
        Flags: D3D12_BUFFER_SRV_FLAG_NONE,
    };
    Some(desc)
}

fn buffer_srv(
    address: u64,
    len: u64,
    desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    buffer_srv_inner(address, len, desc, at, false)
}

fn buffer_srv_inner(
    address: u64,
    len: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    track_shadow: bool,
) -> Option<Kept> {
    let shadow = crate::shadow::find(address, len)?;
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let offset = shadow.offset(address);
    let host = host()?;
    if element == 0 {
        once!(
            offset,
            element,
            "buffer view not on an element of its host copy"
        );
        return None;
    }
    // A view of no elements is an invalid call on the host (it removes the
    // device); the title makes such views for buffers it has not sized yet.
    // The caller makes a null view instead.
    if buffer.NumElements == 0 {
        once!("a buffer view of no elements is left null");
        return None;
    }
    let view_offset = buffer.FirstElement.checked_mul(element)?;
    let bytes = u64::from(buffer.NumElements).checked_mul(element)?;
    if track_shadow && view_offset.checked_add(bytes)? > len {
        return None;
    }
    if offset % element != 0 {
        // The title's placed buffer starts at an address that is not a
        // multiple of `element` inside our larger host shadow buffer (e.g. a
        // non-power-of-two structured element stride inside a ring buffer).
        // Copy the viewed bytes into a persistently mapped aligned upload
        // buffer starting at element 0, and track it so draws refresh it from
        // title memory at submission time.
        const MAX_UNALIGNED_COPY: u64 = 1 << 20;
        if bytes == 0 || bytes > MAX_UNALIGNED_COPY {
            once!(
                offset,
                element,
                "buffer view not on an element of its host copy"
            );
            return None;
        }
        let start = address.checked_add(view_offset)?;
        let (upload, dst) = host.upload_buffer(bytes)?;
        crate::host::copy_mapped(dst, start as *const u8, bytes as usize, false);
        buffer.FirstElement = 0;
        // SAFETY: a live upload buffer and a slot of our heap.
        unsafe {
            host.device
                .CreateShaderResourceView(&upload, Some(&desc), at)
        };
        return Some(Kept::Unaligned(UnalignedCopy {
            resource: upload,
            dst: dst as usize,
            src: start as usize,
            len: bytes as usize,
        }));
    }
    buffer.FirstElement += offset / element;
    if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 && !raw_start_fits(buffer.FirstElement) {
        let first = buffer.FirstElement;
        once!(first, "raw buffer view not on 16 bytes of its host copy");
        return None;
    }
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateShaderResourceView(&shadow.buffer, Some(&desc), at)
    };
    if track_shadow {
        Some(Kept::AliasedShadow {
            shadow,
            address: address.checked_add(view_offset)?,
            len: bytes,
        })
    } else {
        Some(Kept::Resource(shadow.buffer.clone()))
    }
}

/// An unordered access view of title buffer memory at `address` (`len`
/// bytes) on its host copy, as [`buffer_srv`]; false if the copy cannot be
/// written by shaders (upload memory).
/// shortcut: what shaders write stays in the host copy; title memory (and
/// so the CPU, and copies run on the CPU) does not see it.
fn buffer_uav(
    address: u64,
    len: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    let shadow = crate::shadow::find(address, len)?;
    // SAFETY: a live buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        once!("unordered access to a buffer whose host copy shaders cannot write");
        return None;
    }
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    };
    let offset = shadow.offset(address);
    if element == 0 || offset % element != 0 {
        once!(
            offset,
            element,
            "unordered buffer view not on an element of its host copy"
        );
        return None;
    }
    // As in `buffer_srv`.
    if buffer.NumElements == 0 {
        once!("an unordered buffer view of no elements is left null");
        return None;
    }
    buffer.FirstElement += offset / element;
    if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 && !raw_start_fits(buffer.FirstElement) {
        let first = buffer.FirstElement;
        once!(
            first,
            "raw unordered buffer view not on 16 bytes of its host copy"
        );
        return None;
    }
    buffer.CounterOffsetInBytes = 0;
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device.CreateUnorderedAccessView(
            &shadow.buffer,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
    Some(Kept::Resource(shadow.buffer.clone()))
}

/// Whether the host lets views of a typed resource name another format of
/// its family (`CastingFullyTypedFormatSupported`; never SNORM for UNORM,
/// nor FLOAT for another type), which [`fits`] uses for sRGB and linear
/// twins only.
fn casts_typed() -> bool {
    static CASTS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CASTS.get_or_init(|| {
        let Some(host) = host() else { return false };
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS3::default();
        // SAFETY: the query writes its own structure of the given size.
        let asked = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS3,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
        };
        let casts = asked.is_ok() && options.CastingFullyTypedFormatSupported.as_bool();
        if !casts {
            tracing::warn!(
                "the host does not cast typed formats: sRGB views of linear textures (and the reverse) read in the texture's own format"
            );
        }
        casts
    })
}

/// Whether the host can make a shader (or unordered access) view of
/// `texture` in `format` (none: the texture's own). The console allows any
/// view; the host refuses (and removes the device for) views without the
/// resource's permission, of typed depth, and in another format than a
/// typed resource's, but for casts it allows (see [`casts_typed`]). A view
/// of a typed texture in another format of its family takes the texture's
/// format, unless it is the texture's sRGB or linear twin and the host
/// casts: then the hardware converts (or not) as the console does.
/// shortcut: other casts within a family (UINT of UNORM, say) read as the
/// texture's own format.
fn fits(texture: &ID3D12Resource, mut format: Option<&mut DXGI_FORMAT>, unordered: bool) -> bool {
    use crate::resource::{DEPTH_FORMATS, family};
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let allowed = if unordered {
        d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
    } else {
        d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
    };
    let own = d.Format.0 as u32;
    if !allowed || DEPTH_FORMATS.iter().any(|&(typed, _)| typed == own) {
        once!(
            unordered,
            format = own,
            flags = d.Flags.0,
            "view the host does not allow"
        );
        return false;
    }
    // Every family has more than one format, its typeless one first.
    let typeless = family(own) == own && family(own + 1) == own;
    if let Some(format) = format.as_deref_mut()
        && unordered
        && let Some(&(_, linear)) = crate::resource::SRGB
            .iter()
            .find(|&&(s, _)| s == format.0 as u32)
    {
        format.0 = linear as i32;
    }
    let view = format.as_ref().map_or(0, |f| f.0 as u32);
    if view == 0 {
        return !typeless;
    }
    // A texture whose console format PC forbids as a render target or UAV
    // was substituted (`resource::target_format`): views name its host format.
    if view != own && crate::resource::target_format(view) == own {
        if let Some(format) = format {
            *format = d.Format;
        }
        return true;
    }
    if family(view) != family(own) {
        once!(view, own, "view format of another family");
        return false;
    }
    let twin = crate::resource::SRGB
        .iter()
        .any(|&(s, l)| (s, l) == (view, own) || (l, s) == (view, own));
    if let Some(format) = format
        && !typeless
        && !(twin && casts_typed())
    {
        *format = d.Format;
    }
    true
}

/// Makes the host descriptor for `view` at `at`: the host resource it
/// views, or `None` if it cannot.
fn make(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Kept> {
    let host = host()?;
    let device = &host.device;
    match view {
        // make_at and the mirror path write a typed null for this view.
        View::Constant(0, _) => None,
        &View::Constant(address, size) => {
            let shadow = crate::shadow::find(address, u64::from(size))?;
            let gpu = shadow.gpu(address);
            if gpu % 256 != 0 {
                once!(
                    address = format_args!("{address:#x}"),
                    "constant buffer not 256-byte aligned on the host"
                );
                return None;
            }
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                BufferLocation: gpu,
                SizeInBytes: size.next_multiple_of(256),
            };
            // SAFETY: a slot of our heap; the range lies in the host copy.
            unsafe { device.CreateConstantBufferView(Some(&desc), at) };
            Some(Kept::Resource(shadow.buffer.clone()))
        }
        View::Shader(resource, desc) => {
            let resource = *resource as crate::object::This;
            if let Some(texture) = crate::resource::host_resource(resource) {
                let mut desc = desc.as_ref().map(srv_desc);
                if !fits(texture, desc.as_mut().map(|d| &mut d.Format), false) {
                    return None;
                }
                // SAFETY: a live texture and a slot of our heap.
                unsafe {
                    device.CreateShaderResourceView(
                        texture,
                        desc.as_ref().map(std::ptr::from_ref),
                        at,
                    )
                };
                Some(Kept::Resource(texture.clone()))
            } else if let (Some((data, len)), Some(desc)) =
                (crate::resource::buffer_bytes(resource), desc)
            {
                buffer_srv(data as u64, len, srv_desc(desc), at)
            } else {
                None
            }
        }
        View::Unordered(resource, _counter, desc) => {
            let resource = *resource as crate::object::This;
            let Some(texture) = crate::resource::host_resource(resource) else {
                return match (crate::resource::buffer_bytes(resource), desc) {
                    (Some((data, len)), Some(desc)) => {
                        buffer_uav(data as u64, len, uav_desc(desc), at)
                    }
                    _ => None,
                };
            };
            let mut desc = desc.as_ref().map(uav_desc);
            if !fits(texture, desc.as_mut().map(|d| &mut d.Format), true) {
                return None;
            }
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                device.CreateUnorderedAccessView(
                    texture,
                    None::<&ID3D12Resource>,
                    desc.as_ref().map(std::ptr::from_ref),
                    at,
                )
            };
            Some(Kept::Resource(texture.clone()))
        }
        View::Sampler(words) => {
            // SAFETY: 13 words in the host's layout.
            let desc = unsafe {
                std::ptr::from_ref(words)
                    .cast::<D3D12_SAMPLER_DESC>()
                    .read_unaligned()
            };
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(Kept::Empty)
        }
        View::ConsoleSampler(words) => {
            let desc = crate::sampler::console(words);
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(Kept::Empty)
        }
        View::Placed(unordered, resource, desc) => {
            if let Some(view) = placed_texture(view) {
                return make(&view, at);
            }
            const BUFFER: u32 = 1;
            if resource[0] != BUFFER {
                // shortcut: unordered access to a texture the title placed
                // itself is not mirrored; give it a resource as shader
                // views of such textures get when a dispatch writes one.
                once!(
                    dimension = resource[0],
                    "placed unordered texture view not mirrored"
                );
                return None;
            }
            // The resource description's width, after the alignment.
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            if *unordered {
                buffer_uav(address, len, uav_desc(words), at)
            } else {
                buffer_srv(address, len, srv_desc(words), at)
            }
        }
    }
}

/// A shader view of a texture the title placed in memory itself, as a view
/// of the resource made for that memory (on first use).
pub(crate) fn placed_texture(view: &View) -> Option<View> {
    const BUFFER: u32 = 1;
    let View::Placed(false, resource, desc) = view else {
        return None;
    };
    if resource[0] == BUFFER {
        return None;
    }
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let texture = crate::resource::texture_at(address, resource)?;
    Some(View::Shader(texture as usize, desc[..10].try_into().ok()))
}

/// A null view of the kind `view` is, for a slot whose view the host
/// cannot make: unordered access (to a buffer or a 2D texture) or a shader
/// view. Writes to it are dropped, reads give zeros.
fn null_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    // Unordered views: whether a buffer, and the title's view dimension.
    let (buffer, dimension) = match view {
        View::Unordered(r, _, desc) => (
            is_buffer(r) || desc.is_some_and(|d| d[1] == 1),
            desc.map(|d| d[1]),
        ),
        View::Placed(true, resource, desc) => (resource[0] == 1, Some(desc[1])),
        // Shader views of buffers read nothing through a null buffer view:
        // a null 2D view where the shader expects a buffer is a mismatch
        // the hardware does not survive in general (GPU-based validation
        // reports it).
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => {
            return null_buffer_view(at);
        }
        View::Placed(false, resource, _) if resource[0] == 1 => return null_buffer_view(at),
        // Textures: a null view of the title's dimension (1D, arrays, 3D,
        // cubes), not always 2D.
        View::Shader(_, Some(desc)) => return null_like(desc, at),
        View::Placed(false, _, desc) => {
            return null_like(desc[..10].try_into().unwrap_or(&[0; 10]), at);
        }
        View::Constant(..) => {
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC::default();
            // SAFETY: a slot of our heap; a zero location makes a null view.
            return unsafe { host.device.CreateConstantBufferView(Some(&desc), at) };
        }
        _ => return null_view(at),
    };
    let desc = if buffer {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            ..Default::default()
        }
    } else {
        null_uav_desc(dimension)
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A null texture UAV of the title's view dimension (1D, 1D or 2D arrays,
/// 3D), 2D if it names none: a 2D null view where the shader writes a 3D
/// texture is a mismatch GPU-based validation reports (the compute shader
/// that copies the world's wind field, 2026-11-24).
fn null_uav_desc(dimension: Option<u32>) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    match dimension.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE1D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                WSize: u32::MAX,
                ..Default::default()
            };
        }
        _ => {}
    }
    desc
}

/// A buffer view that reads zeros (raw, as translated shaders mostly read
/// buffers).
fn null_buffer_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A texture view that reads zeros with the dimension (and mip and array
/// counts) of the title's description `words`; 2D if it names none.
fn null_like(words: &[u32; 10], at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    const TEXTURES: std::ops::RangeInclusive<u32> = 2..=10;
    if !TEXTURES.contains(&words[1]) {
        return null_view(at);
    }
    let Some(host) = host() else { return };
    // One mip, one slice or cube, plane 0: fields a null view still checks.
    let mut w = [0u32; 10];
    w[1] = words[1];
    match words[1] {
        6 => {}
        7 => w[5] = 1,
        3 | 5 | 10 => (w[5], w[7]) = (1, 1),
        _ => w[5] = 1,
    }
    let mut desc = srv_desc(&w);
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A view that reads zeros, for slots whose view the host cannot make.
/// shortcut: slots with no view at all (kind unknown) get this null 2D
/// view; null buffer views there did not stop the world's low-address
/// page faults (2026-11-24, run w10), so they are not the cause.
fn null_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The CPU-side block used to initialize null samplers must fit the
    /// same portable sampler limit as the shader-visible heap.
    #[test]
    fn portable_sampler_null_fill_fits_the_host_limit() {
        if host().is_none() {
            return; // no host GPU
        }
        let heap = HostHeap::new(true, D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE)
            .expect("portable sampler heap must initialize");
        assert!(
            heap.fill_null(
                D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE,
            )
            .is_ok()
        );
    }

    /// A null texture UAV keeps the title's dimension (3D for the world's
    /// wind field), and is 2D when the title names none.
    #[test]
    fn null_uavs_keep_the_titles_dimension() {
        let d = null_uav_desc(Some(8));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture3D.WSize }, u32::MAX);
        let d = null_uav_desc(Some(5));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture2DArray.ArraySize }, 1);
        let dimension = |d| null_uav_desc(d).ViewDimension;
        assert_eq!(dimension(Some(2)), D3D12_UAV_DIMENSION_TEXTURE1D);
        assert_eq!(dimension(Some(4)), D3D12_UAV_DIMENSION_TEXTURE2D);
        assert_eq!(dimension(None), D3D12_UAV_DIMENSION_TEXTURE2D);
        // Multisampled UAVs need a cap; they stay 2D.
        assert_eq!(dimension(Some(6)), D3D12_UAV_DIMENSION_TEXTURE2D);
    }

    /// A null 3D UAV made on the real device leaves it alive.
    #[test]
    fn null_3d_uav_is_accepted_by_the_device() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let view = View::Unordered(0, 0, Some([28, 8, 0, 0, 0, 0, 0, 0, 0, 0]));
        null_for(&view, heap.cpu(0));
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Buffer views of no elements (the title makes them for buffers it has
    /// not sized yet) are left null: the host removes the device for one.
    #[test]
    fn buffer_views_of_no_elements_are_left_null() {
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_UINT;
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let memory = vec![0u32; 1024];
        let address = memory.as_ptr() as u64;
        let resource = 0x7e57_0000usize;
        crate::shadow::register(resource, address, 4096);
        let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV::default(),
            },
        };
        assert!(buffer_uav(address, 4096, uav, heap.cpu(0)).is_none());
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV::default(),
            },
        };
        assert!(buffer_srv(address, 4096, srv, heap.cpu(1)).is_none());
        crate::shadow::unregister(resource, address);
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Dynamic slots stay taken until given back, and runs do not wrap.
    #[test]
    fn dynamic_slots_are_held_until_released() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let a = heap.dynamic(heap.spare - 100).unwrap();
        let b = heap.dynamic(60).unwrap();
        assert_eq!((a, b), (4, 4 + heap.spare - 100));
        // 40 left at the end: 50 do not fit until something is given back.
        assert_eq!(heap.dynamic(50), None);
        heap.release(a, 50);
        assert_eq!(heap.dynamic(50), Some(4));
        assert_eq!(heap.dynamic(40), Some(4 + heap.spare - 40));
    }

    /// A sampler heap past the host's limit is capped, not lost, and keeps
    /// dynamic slots for the tables reaching past what it holds.
    #[test]
    fn large_sampler_heap_is_capped() {
        let Some(heap) = HostHeap::new(true, 4 * SAMPLERS) else {
            return; // no host GPU
        };
        let count = SAMPLERS - SAMPLER_DYNAMIC;
        assert_eq!(heap.count(), count);
        assert_eq!(heap.dynamic(SAMPLER_DYNAMIC + 1), None);
        let a = heap.dynamic(SAMPLER_DYNAMIC).unwrap();
        assert_eq!(a, count);
        assert_eq!(heap.dynamic(1), None);
        heap.release(a, 8);
        assert_eq!(heap.dynamic(8), Some(count));
    }

    /// A sampler heap within the limit has a slot per title slot and no
    /// dynamic ones.
    #[test]
    fn small_sampler_heap_has_no_dynamic_slots() {
        let Some(heap) = HostHeap::new(true, 16) else {
            return; // no host GPU
        };
        assert_eq!(heap.count(), 16);
        assert_eq!(heap.dynamic(1), None);
    }

    /// A structured buffer SRV whose offset inside its shadow buffer is not a
    /// multiple of its element stride is copied into an aligned upload buffer
    /// and refreshed from title memory when a command list referencing the slot
    /// is submitted.
    #[test]
    fn unaligned_structured_buffer_srv_refreshes_on_submission() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let mut backing: Vec<u8> = (0..4096usize).map(|i| (i % 251) as u8).collect();
        let base = backing.as_ptr() as u64;
        crate::shadow::register(base as usize, base, backing.len() as u64);
        // Offset 32 is not a multiple of stride 24 (32 % 24 == 8).
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: 0,
                    NumElements: 1,
                    StructureByteStride: 24,
                    Flags: D3D12_BUFFER_SRV_FLAG_NONE,
                },
            },
        };
        let kept = buffer_srv(base + 32, 24, desc, heap.cpu(0))
            .expect("unaligned structured SRV should succeed");
        let (upload, dst) = match &kept {
            Kept::Unaligned(copy) => (copy.resource.clone(), copy.dst as *const u8),
            _ => panic!("expected an unaligned upload copy"),
        };
        heap.keep(0, kept);

        // Mutate the title slice *after* creating the view (as a ring buffer does).
        let updated: [u8; 24] = std::array::from_fn(|i| 0xA0u8.wrapping_add(i as u8));
        backing[32..56].copy_from_slice(&updated);

        let list = host.new_list().expect("host command list");
        heap.refresh_slot(0, &list);
        list.close();
        host.execute(&[&list]);
        crate::shadow::unregister(base as usize, base);

        // SAFETY: `upload` is persistently mapped for its lifetime (`24` bytes)
        // and `host.execute` has completed.
        unsafe {
            assert_eq!(upload.GetDesc().Width, 24);
            let got = std::slice::from_raw_parts(dst, 24);
            assert_eq!(got, &updated);
            assert!(host.device.GetDeviceRemovedReason().is_ok());
        }
    }
}

#[cfg(test)]
mod structured_alias_tests {
    use super::*;
    use crate::descriptor::{create_placed_srv, view};

    fn raw_srv_desc(first_dword: u64, dwords: u32) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
        D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: first_dword,
                    NumElements: dwords,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        }
    }

    fn placed_raw_srv_handle(
        slot: &mut [u64; 16],
        address: u64,
        len: u64,
        first_dword: u32,
        dwords: u32,
    ) -> u64 {
        let mut resource = [0u32; 14];
        resource[0] = 1; // BUFFER
        resource[4] = len as u32;
        resource[5] = (len >> 32) as u32;
        let srv = raw_srv_desc(u64::from(first_dword), dwords);
        // SAFETY: `D3D12_SHADER_RESOURCE_VIEW_DESC` is 40 bytes (`[u32; 10]`).
        let srv_words: [u32; 10] = unsafe {
            std::ptr::from_ref(&srv)
                .cast::<[u32; 10]>()
                .read_unaligned()
        };
        let mut desc = [0u32; 16];
        desc[..10].copy_from_slice(&srv_words);
        desc[10] = address as u32;
        desc[11] = (address >> 32) as u32;
        let handle = slot.as_mut_ptr() as u64;
        create_placed_srv(std::ptr::null(), &resource, &desc, handle);
        handle
    }

    struct ShadowGuard {
        resource: usize,
        address: u64,
    }

    impl Drop for ShadowGuard {
        fn drop(&mut self) {
            crate::shadow::unregister(self.resource, self.address);
        }
    }

    #[test]
    fn structured_alias_preserves_exact_byte_bounds() {
        let alias = structured_alias(raw_srv_desc(16, 32), 64).expect("64-byte aligned raw SRV");
        assert_eq!(alias.Format, DXGI_FORMAT_UNKNOWN);
        assert_eq!(alias.ViewDimension, D3D12_SRV_DIMENSION_BUFFER);
        // SAFETY: `ViewDimension == D3D12_SRV_DIMENSION_BUFFER`.
        let buf = unsafe { alias.Anonymous.Buffer };
        assert_eq!(buf.FirstElement, 1);
        assert_eq!(buf.NumElements, 2);
        assert_eq!(buf.StructureByteStride, 64);
        assert_eq!(buf.Flags, D3D12_BUFFER_SRV_FLAG_NONE);
    }

    #[test]
    fn structured_alias_rejects_fractional_typed_structured_and_non_buffer_views() {
        // Fractional byte offset (4 bytes % 64 != 0).
        assert!(structured_alias(raw_srv_desc(1, 32), 64).is_none());
        // Fractional byte length (124 bytes % 64 != 0).
        assert!(structured_alias(raw_srv_desc(16, 31), 64).is_none());
        // Zero elements.
        assert!(structured_alias(raw_srv_desc(0, 0), 64).is_none());
        // Already-structured SRV must not be reinterpreted.
        let mut structured = raw_srv_desc(0, 2);
        structured.Format = DXGI_FORMAT_UNKNOWN;
        structured.Anonymous.Buffer.StructureByteStride = 64;
        structured.Anonymous.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
        assert!(structured_alias(structured, 64).is_none());
        // Typed non-raw buffer SRV must not be reinterpreted.
        let typed = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: 0,
                    NumElements: 8,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_NONE,
                },
            },
        };
        assert!(structured_alias(typed, 64).is_none());
        // Non-buffer (texture) SRV must not be aliased.
        let tex = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            ..Default::default()
        };
        assert!(structured_alias(tex, 64).is_none());
        // Invalid strides.
        assert!(structured_alias(raw_srv_desc(0, 16), 0).is_none());
        assert!(structured_alias(raw_srv_desc(0, 16), 6).is_none());
        assert!(structured_alias(raw_srv_desc(0, 16), 4096).is_none());
    }

    #[test]
    fn make_structured_at_rejects_range_past_placed_buffer() {
        let Some(_) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let backing = vec![0u8; 1024];
        let base = backing.as_ptr() as u64;
        let _guard = ShadowGuard {
            resource: base as usize,
            address: base,
        };
        crate::shadow::register(base as usize, base, backing.len() as u64);

        // Placed length is 128 bytes, but view reads FirstElement=16 (64B) + 128B = 192B > 128B.
        let mut slot = [0u64; 16];
        let handle = placed_raw_srv_handle(&mut slot, base + 64, 128, 16, 32);
        let v = view(handle).expect("placed raw SRV");
        let dyn_idx = heap.dynamic(1).expect("dynamic slot");
        assert!(!heap.make_structured_at(dyn_idx, &v, 64));
    }

    #[test]
    fn structured_alias_reads_exact_records_with_sentinel_and_mutation_checks() {
        use windows::Win32::Foundation::CloseHandle;
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
        use windows::core::{Interface, s};

        let Some(host) = host() else {
            return; // no host GPU
        };
        const SOURCE: &str = r#"
#define RS "DescriptorTable(SRV(t0, numDescriptors=1)), DescriptorTable(SRV(t1, numDescriptors=1)), UAV(u0)"
struct Record { uint w[16]; };
StructuredBuffer<Record> aliased : register(t0);
ByteAddressBuffer canonical_raw : register(t1);
RWByteAddressBuffer dst : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint count = 0;
    uint stride = 0;
    aliased.GetDimensions(count, stride);
    dst.Store(0, count);
    dst.Store(4, stride);
    for (uint r = 0; r < 2; ++r) {
        Record rec = aliased[r];
        for (uint i = 0; i < 16; ++i) {
            dst.Store(8 + (r * 16 + i) * 4, rec.w[i]);
        }
    }
    dst.Store(136, canonical_raw.Load(0));
}
"#;
        let mut blob: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        // SAFETY: source string and entry names are static and valid.
        unsafe {
            D3DCompile(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                s!("structured_alias_readback"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut blob,
                Some(&mut errors),
            )
            .unwrap_or_else(|e| {
                let msg = errors
                    .as_ref()
                    .map(|b| {
                        let p = b.GetBufferPointer().cast::<u8>();
                        String::from_utf8_lossy(std::slice::from_raw_parts(p, b.GetBufferSize()))
                            .into_owned()
                    })
                    .unwrap_or_default();
                panic!("D3DCompile failed: {e}: {msg}");
            });
        }
        let blob = blob.unwrap();
        // SAFETY: `blob` owns compiled DXBC bytes.
        let code = unsafe {
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
        };
        // SAFETY: `code` contains embedded root signature and compute shader.
        let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
            let root = host.device.CreateRootSignature(0, code).unwrap();
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: code.as_ptr().cast(),
                    BytecodeLength: code.len(),
                },
                ..Default::default()
            };
            let pso = host.device.CreateComputePipelineState(&desc).unwrap();
            (root, pso)
        };

        let make_buffer = |kind: D3D12_HEAP_TYPE, flags, state| -> ID3D12Resource {
            let mut b: Option<ID3D12Resource> = None;
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 512,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: flags,
                ..Default::default()
            };
            let heap = D3D12_HEAP_PROPERTIES {
                Type: kind,
                ..Default::default()
            };
            // SAFETY: valid buffer creation parameters.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        state,
                        None,
                        &mut b,
                    )
                    .unwrap();
            }
            b.unwrap()
        };

        let out = make_buffer(
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        );
        let readback = make_buffer(
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );

        let mut backing = vec![0xCCu8; 4096];
        let base = backing.as_ptr() as u64;
        let _guard = ShadowGuard {
            resource: base as usize,
            address: base,
        };
        crate::shadow::register(base as usize, base, backing.len() as u64);

        let write_word = |buf: &mut [u8], byte_off: usize, val: u32| {
            buf[byte_off..byte_off + 4].copy_from_slice(&val.to_le_bytes());
        };
        let fill_sentinels = |buf: &mut [u8], start: usize, len: usize, pat: u8| {
            buf[start..start + len].fill(pat);
        };

        // Case 1: 64-byte aligned start inside shadow (`shadow.offset == 64`).
        // Case 2: 16-byte aligned (raw-valid) but unaligned for stride 64 (`shadow.offset == 528`).
        for (tag, placed_off) in [(0x1000_0000u32, 64usize), (0x2000_0000u32, 528usize)] {
            // 64B prefix sentinel + 128B (2 records) + 64B suffix sentinel = 256B.
            let prefix_off = placed_off;
            let rec_off = placed_off + 64;
            let suffix_off = placed_off + 192;
            fill_sentinels(&mut backing, prefix_off, 64, 0xA5);
            for i in 0..32u32 {
                write_word(&mut backing, rec_off + i as usize * 4, tag | i);
            }
            fill_sentinels(&mut backing, suffix_off, 64, 0x5A);

            let mut slot = [0u64; 16];
            let handle = placed_raw_srv_handle(&mut slot, base + placed_off as u64, 192, 16, 32);
            let heap = HostHeap::new(false, 4).expect("host heap");
            let v = heap
                .mirror(0, handle, true)
                .expect("canonical slot 0 raw mirror");
            let dyn_idx = heap.dynamic(1).expect("dynamic alias slot");
            assert!(
                heap.make_structured_at(dyn_idx, &v, 64),
                "make_structured_at failed for offset {placed_off}"
            );

            // Mutate dword 17 after creating the alias to prove submission-time
            // refresh updates the backing resource (shadow or UnalignedCopy).
            let mutated_word = tag | 0x00FF_0011;
            write_word(&mut backing, rec_off + 17 * 4, mutated_word);

            let list = host.new_list().expect("host list");
            assert!(heap.refresh_slot(dyn_idx, &list));
            if placed_off % 64 != 0 {
                // `dyn_idx` reads an `UnalignedCopy` upload buffer rather than `shadow`,
                // so refresh `canonical_raw`'s first record word independently to check
                // canonical slot 0's raw readback too.
                crate::shadow::find(base + rec_off as u64, 4)
                    .expect("registered shadow")
                    .refresh(&list, base + rec_off as u64, 4);
            }
            // SAFETY: open list with live compute root signature, PSO, and heaps.
            unsafe {
                let l = &list.list;
                l.SetComputeRootSignature(&root);
                l.SetPipelineState(&pso);
                l.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
                l.SetComputeRootDescriptorTable(0, heap.gpu(dyn_idx));
                l.SetComputeRootDescriptorTable(1, heap.gpu(0));
                l.SetComputeRootUnorderedAccessView(2, out.GetGPUVirtualAddress());
                l.Dispatch(1, 1, 1);
                l.ResourceBarrier(&[crate::host::transition(
                    &out,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                l.CopyBufferRegion(&readback, 0, &out, 0, 140);
                l.ResourceBarrier(&[crate::host::transition(
                    &out,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
            }
            list.close();
            host.execute(&[&list]);

            // Validate descriptor binding cleanly through a standalone queue when
            // the D3D12 InfoQueue is active (`VICTORY_D3D12_DEBUG=1`) so we inspect
            // messages before `host.execute` clears them.
            if let Ok(iq) = host.device.cast::<ID3D12InfoQueue>() {
                // SAFETY: standalone queue/allocator/list/fence on `host.device`.
                unsafe {
                    let queue: ID3D12CommandQueue = host
                        .device
                        .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                            ..Default::default()
                        })
                        .unwrap();
                    let alloc: ID3D12CommandAllocator = host
                        .device
                        .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                        .unwrap();
                    let cl: ID3D12GraphicsCommandList = host
                        .device
                        .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
                        .unwrap();
                    cl.SetComputeRootSignature(&root);
                    cl.SetPipelineState(&pso);
                    cl.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
                    cl.SetComputeRootDescriptorTable(0, heap.gpu(dyn_idx));
                    cl.SetComputeRootDescriptorTable(1, heap.gpu(0));
                    cl.SetComputeRootUnorderedAccessView(2, out.GetGPUVirtualAddress());
                    cl.Dispatch(1, 1, 1);
                    cl.Close().unwrap();

                    let before = iq.GetNumStoredMessages();
                    queue.ExecuteCommandLists(&[Some(cl.into())]);
                    let fence: ID3D12Fence =
                        host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
                    let event = CreateEventW(None, false, false, None).unwrap();
                    queue.Signal(&fence, 1).unwrap();
                    fence.SetEventOnCompletion(1, event).unwrap();
                    WaitForSingleObject(event, INFINITE);
                    let _ = CloseHandle(event);

                    let after = iq.GetNumStoredMessages();
                    for idx in before.min(after)..after {
                        let mut size = 0usize;
                        if iq.GetMessage(idx, None, &mut size).is_err() || size == 0 {
                            continue;
                        }
                        let mut raw = vec![0u64; size.div_ceil(8)];
                        let msg = raw.as_mut_ptr().cast::<D3D12_MESSAGE>();
                        if iq.GetMessage(idx, Some(msg), &mut size).is_ok() {
                            let m = &*msg;
                            let desc =
                                std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                            let text = String::from_utf8_lossy(desc);
                            assert_ne!(m.ID.0, 1387, "unexpected ID 1387: {text}");
                            assert!(
                                m.Severity.0 > D3D12_MESSAGE_SEVERITY_ERROR.0,
                                "unexpected D3D12 error {}: {text}",
                                m.ID.0
                            );
                        }
                    }
                }
            }

            let mut mapped = std::ptr::null_mut();
            let mut dwords = [0u32; 35];
            // SAFETY: execution completed and readback holds 140 bytes (35 dwords).
            unsafe {
                let range = D3D12_RANGE { Begin: 0, End: 140 };
                readback.Map(0, Some(&range), Some(&mut mapped)).unwrap();
                std::ptr::copy_nonoverlapping(mapped.cast::<u32>(), dwords.as_mut_ptr(), 35);
                readback.Unmap(0, None);
            }
            assert_eq!(dwords[0], 2, "GetDimensions count for offset {placed_off}");
            assert_eq!(
                dwords[1], 64,
                "GetDimensions stride for offset {placed_off}"
            );
            for i in 0..32u32 {
                let expected = if i == 17 { mutated_word } else { tag | i };
                assert_eq!(
                    dwords[2 + i as usize],
                    expected,
                    "record dword {i} mismatch for offset {placed_off}"
                );
            }
            assert_eq!(
                dwords[34], tag,
                "canonical slot 0 must stay a valid raw SRV for offset {placed_off}"
            );
            assert!(
                backing[prefix_off..prefix_off + 64]
                    .iter()
                    .all(|&b| b == 0xA5)
            );
            assert!(
                backing[suffix_off..suffix_off + 64]
                    .iter()
                    .all(|&b| b == 0x5A)
            );
        }

        // SAFETY: plain device health query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }
}
