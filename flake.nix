{
  description = "PyTorch with cuda";

  inputs = {
    # It is unrelated to the URL of nixpkgs, any version can be used.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:

    let
      pkgs = import nixpkgs {
        system = "x86_64-linux";
        config.allowUnfree = true;
      };
    in
    {
      devShells."x86_64-linux".default = pkgs.mkShell {
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc
          "/run/opengl-driver"
        ];
        venvDir = ".venv";
        packages = with pkgs; [
          python312
          # Do not install PyTorch with cuda directly here!!!
          python312Packages.venvShellHook
          python312Packages.pip
          python312Packages.pysocks
        ];
      };
    };
}
