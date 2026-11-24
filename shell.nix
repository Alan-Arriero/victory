{
  # Legacy nix-shell uses exactly the same inputs as nix develop.
  pkgs ?
    let
      lock = builtins.fromJSON (builtins.readFile ./flake.lock);
      source =
        name:
        let
          inherit (lock.nodes.${name}.locked)
            owner
            repo
            rev
            narHash
            ;
        in
        builtins.fetchTarball {
          url = "https://github.com/${owner}/${repo}/archive/${rev}.tar.gz";
          sha256 = narHash;
        };
    in
    import (source "nixpkgs") {
      overlays = [ (import (source "rust-overlay")) ];
    },
  withProton ? false,
}:
let
  rust = (builtins.fromTOML (builtins.readFile ./rust-toolchain.toml)).toolchain;
  toolchain = pkgs.rust-bin.fromRustupToolchain (
    rust
    // {
      components = rust.components ++ [
        "rust-src"
        "rust-analyzer"
      ];
    }
  );
in
assert pkgs.stdenv.hostPlatform.system == "x86_64-linux";
pkgs.mkShell {
  name = "victory-linux";
  packages = [
    toolchain
    pkgs.cargo-xwin
    # Use the unwrapped compiler for MSVC: the Linux wrapper injects host flags.
    pkgs.llvmPackages.clang-unwrapped
    pkgs.llvmPackages.lld
    pkgs.llvmPackages.llvm
    pkgs.directx-shader-compiler
    pkgs.pkg-config
    pkgs.git
    pkgs.shellcheck
    pkgs.nixfmt
  ]
  ++ pkgs.lib.optionals withProton [
    pkgs.umu-launcher
    pkgs.vulkan-tools
  ];

  shellHook = ''
    # Compile the fallback shaders into Windows builds on the Linux host.
    export VICTORY_BUILD_DXC="''${VICTORY_BUILD_DXC:-${pkgs.directx-shader-compiler}/bin/dxc}"
    export XWIN_ARCH="''${XWIN_ARCH:-x86_64}"

    # XDG bases must be absolute. Keep SDK downloads out of the checkout.
    victory_cache_home="''${XDG_CACHE_HOME:-$HOME/.cache}"
    case "$victory_cache_home" in
      /*) ;;
      *) victory_cache_home="$HOME/.cache" ;;
    esac
    export XWIN_CACHE_DIR="''${XWIN_CACHE_DIR:-$victory_cache_home/cargo-xwin}"
    unset victory_cache_home
  '';
}
