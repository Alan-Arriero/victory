{
  description = "VIctory Linux development and Windows cross-compilation";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { nixpkgs, rust-overlay, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        overlays = [ rust-overlay.overlays.default ];
      };
    in
    {
      devShells.${system} = {
        default = import ./shell.nix { inherit pkgs; };
        proton = import ./shell.nix {
          inherit pkgs;
          withProton = true;
        };
      };

      formatter.${system} = pkgs.nixfmt;

      checks.${system}.launcher =
        pkgs.runCommand "victory-launcher-check"
          {
            nativeBuildInputs = [
              pkgs.bash
              pkgs.shellcheck
            ];
          }
          ''
            shellcheck ${./scripts/victory-proton.sh} ${./tests/proton-launcher.sh}
            bash ${./tests/proton-launcher.sh} ${./scripts/victory-proton.sh}
            touch "$out"
          '';
    };
}
