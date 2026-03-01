{
  description = "Podly development flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python311
            python311Packages.pip
            pipenv
            ffmpeg
            git
            pkg-config
          ];

          shellHook = ''
            export PIPENV_VENV_IN_PROJECT=1
            echo "Podly nix shell ready. Run: pipenv install --dev"
          '';
        };
      });
}
