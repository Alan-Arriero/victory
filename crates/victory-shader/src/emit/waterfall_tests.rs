//! End-to-end tests of waterfall loops over per-lane descriptors: the
//! synthetic GFX10 programs of `resource::tests` lowered to DXIL, validated
//! by `dxil.dll` when present, and run on the host GPU and WARP, each lane
//! reading the resource its own descriptor names.

use windows::Win32::Graphics::Direct3D12::*;

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// Lowers a wave32 compute program of 32 threads with the table of views at
/// s0 (textures, or with `buffers` structured buffers of dwords, at entries
/// 0 and 1, a raw UAV at entry 2) and the table of samplers at s1.
fn build(words: &[u32], buffers: bool) -> HostShader {
    let (shader, lowered) = build_any(words, buffers);
    // One access with each lane's own heap index, no wave operation.
    assert_eq!(lowered.nonuniform_handles, 1);
    assert!(!lowered.wave_ops);
    shader
}

/// [`build`] without its checks: the shader (validated by `dxil.dll` when
/// present) and what lowering reported.
fn build_any(words: &[u32], buffers: bool) -> (HostShader, lower::Lowered) {
    let table = |user_data: u8| RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot: user_data,
    };
    let range = |kind, table_offset_dwords, dwords, stride, user_data| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords,
        user_data,
        slot: user_data,
    };
    let views = if buffers {
        range(ResourceType::Srv, 0, 4, 4, 0)
    } else {
        range(ResourceType::Srv, 0, 8, 0, 0)
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: vec![table(0), table(1)],
        used: vec![
            views,
            range(ResourceType::Uav, 16, 4, 0, 0),
            range(ResourceType::Sampler, 0, 4, 0, 1),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .expect("lower");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil.clone(),
        user_data: lowered.user_data.clone(),
        samplers: lowered.samplers.clone(),
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs.clone(),
        root_srvs: lowered.root_srvs.clone(),
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    (shader, lowered)
}

/// Runs `shader` with `views` at heap indices 0.. (the UAV last) on the
/// hardware GPU and on WARP (each skipped without a device) and returns,
/// per one that ran, the UAV's 32 dwords.
fn run(shader: &HostShader, views: &[View]) -> Vec<(&'static str, Vec<u32>)> {
    let samplers = [D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        MaxLOD: f32::MAX,
        ..Default::default()
    }];
    let mut out = Vec::new();
    // One GPU test at a time, with the VALU tests: in parallel on the
    // hardware adapter and WARP the test process sometimes died of an
    // access violation (7 of 24 runs of these four tests).
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = match host.compute_pipeline(shader) {
            Ok(p) => p,
            Err(e) => panic!("{label}: compute_pipeline failed: {e}"),
        };
        // Both tables start at heap index 0.
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            assert_eq!(u.root, RootType::Table, "{u:?}");
            ud[u.index as usize] = 0;
        }
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views,
                    samplers: &samplers,
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push((label, words[0].clone()));
    }
    out
}

/// The red channel each lane sampled from two 1x1 textures, red 1.0 at
/// heap index 0 and 2.0 at 1.
fn sampled(shader: &HostShader) -> Vec<(&'static str, Vec<f32>)> {
    let first = [[1.0f32, 0.0, 0.0, 1.0]];
    let second = [[2.0f32, 0.0, 0.0, 1.0]];
    let views = [
        View::Texture(1, 1, &first),
        View::Texture(1, 1, &second),
        View::RawUav(32),
    ];
    run(shader, &views)
        .into_iter()
        .map(|(label, words)| (label, words.into_iter().map(f32::from_bits).collect()))
        .collect()
}

#[test]
fn lanes_sample_the_texture_their_if_else_arm_chose() {
    use crate::resource::tests::WATERFALL_JOINED_WAVE32;
    let shader = build(WATERFALL_JOINED_WAVE32, false);
    for (label, red) in sampled(&shader) {
        // Lanes below 16 took the first texture, the others the second.
        let want: Vec<f32> = (0..32).map(|l| if l < 16 { 1.0 } else { 2.0 }).collect();
        assert_eq!(red, want, "{label}");
    }
}

#[test]
fn lanes_sample_the_texture_their_arm_loaded_or_picked() {
    use crate::resource::tests::WATERFALL_JOINED_MIXED_WAVE32;
    let shader = build(WATERFALL_JOINED_MIXED_WAVE32, false);
    for (label, red) in sampled(&shader) {
        // Lanes below 16 loaded entry (lane & 1); of the others, lanes
        // below 24 picked the first, the rest the second.
        let want: Vec<f32> = (0..32)
            .map(|l| match l {
                0..16 => [1.0, 2.0][l & 1],
                16..24 => 1.0,
                _ => 2.0,
            })
            .collect();
        assert_eq!(red, want, "{label}");
    }
}

#[test]
fn lanes_load_from_the_buffer_their_nested_arms_chose() {
    use crate::resource::tests::WATERFALL_JOINED_BUFFER_WAVE32;
    let shader = build(WATERFALL_JOINED_BUFFER_WAVE32, true);
    let first: Vec<u32> = (0..32).map(|i| 100 + i).collect();
    let second: Vec<u32> = (0..32).map(|i| 200 + i).collect();
    let views = [
        View::Buffer {
            data: &first,
            stride: Stride::Structured(4),
        },
        View::Buffer {
            data: &second,
            stride: Stride::Structured(4),
        },
        View::RawUav(32),
    ];
    for (label, got) in run(&shader, &views) {
        // Dword `lane` of the first buffer below 16, of the second from 16
        // to 23; lanes from 24 on load and store nothing.
        let want: Vec<u32> = (0..32)
            .map(|l| match l {
                0..16 => 100 + l,
                16..24 => 200 + l,
                _ => 0,
            })
            .collect();
        assert_eq!(got, want, "{label}");
    }
}

#[test]
fn lanes_sample_the_entry_their_offset_names_in_the_console_waterfall() {
    use crate::resource::tests::WATERFALL_INDEXED_WAVE32;
    let shader = build(WATERFALL_INDEXED_WAVE32, false);
    let results = sampled(&shader);
    assert!(!results.is_empty());
    for (label, red) in results {
        // Even lanes hold entry 0's offset, odd lanes entry 1's.
        let want: Vec<f32> = (0..32).map(|l| [1.0, 2.0][l & 1]).collect();
        assert_eq!(red, want, "{label}");
    }
}

#[test]
fn lanes_sample_their_own_texture_in_a_waterfall_narrowed_by_v_cmpx() {
    use crate::resource::tests::cmpx_waterfall;
    // Lanes below 16 hold the first descriptor, the others the second.
    let want: Vec<f32> = (0..32).map(|l| if l < 16 { 1.0 } else { 2.0 }).collect();
    for folded in [false, true] {
        let shader = build(&cmpx_waterfall(false, folded), false);
        let results = sampled(&shader);
        assert!(!results.is_empty());
        for (label, red) in results {
            assert_eq!(red, want, "{label}, folded {folded}");
        }
    }
}

/// `image_gather4_l` at (0.375, 0.375) of a 4x4 texture whose red is
/// 10 * y + x, level 2.0 (the view has one level: clamped to level 0). Lanes
/// 0-7 store the four results at dwords 4 * lane...
const GATHER4_L_WAVE32: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xBE86_0301, // s_mov_b32 s6, s1
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF40C_0302,
    0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
    0xF408_0203,
    0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
    0xF408_0A02,
    0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
    0x3620_0087, // v_and_b32 v16, 7, v0
    0x3420_2084, // v_lshlrev_b32 v16, 4, v16
    0x7E04_02FF,
    0x3EC0_0000, // v_mov_b32 v2, 0.375
    0x7E06_02FF,
    0x3EC0_0000, // v_mov_b32 v3, 0.375
    0x7E08_02FF,
    0x4000_0000, // v_mov_b32 v4, 2.0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xF110_0108,
    0x0043_0C02, // image_gather4_l v[12:15], v[2:4], s[12:19], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_2D
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xE070_1000,
    0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
    0xE070_1004,
    0x800A_0D10, // buffer_store_dword v13, v16, s[40:43], 0 offen offset:4
    0xE070_1008,
    0x800A_0E10, // buffer_store_dword v14, v16, s[40:43], 0 offen offset:8
    0xE070_100C,
    0x800A_0F10, // buffer_store_dword v15, v16, s[40:43], 0 offen offset:12
    0xBF81_0000, // s_endpgm
];

#[test]
fn gather4_with_a_level_returns_the_four_texels_of_that_level_in_gather_order() {
    let (shader, _) = build_any(GATHER4_L_WAVE32, false);
    let texels: Vec<[f32; 4]> = (0..16)
        .map(|i| [(10 * (i / 4) + i % 4) as f32, 0.0, 0.0, 1.0])
        .collect();
    let views = [
        View::Texture(4, 4, &texels),
        View::Texture(4, 4, &texels),
        View::RawUav(32),
    ];
    let results = run(&shader, &views);
    assert!(!results.is_empty());
    for (label, words) in results {
        let got: Vec<f32> = words.into_iter().map(f32::from_bits).collect();
        // u * 4 - 0.5 = 1: texels (1, 2), (2, 2), (2, 1), (1, 1).
        let want: Vec<f32> = (0..32).map(|i| [21.0, 22.0, 12.0, 11.0][i % 4]).collect();
        assert_eq!(got, want, "{label}");
    }
}
