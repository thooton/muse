let
  nixpkgs-src = builtins.fetchTarball {
    # Master of Dec 21, 2023
    url = "https://github.com/NixOS/nixpkgs/archive/08b802c343d93f4d09deeebf187f6ec8c3233124.tar.gz";
    sha256 = "0zx0d52cpjr0nxfd3w2qzaaqbq83pj1iqw8300hsij9mhxlny3vw";
  };

  pkgs = import nixpkgs-src {
    config = {
      allowUnfree = true;
    };
  };

  myPython = pkgs.python311;

  pythonWithPkgs = myPython.withPackages (pythonPkgs:
    with pythonPkgs; [
      black # for formatting
      pip
      setuptools
      virtualenvwrapper
      wheel
    ]);

  lib-path = with pkgs;
    lib.makeLibraryPath [
      libffi
      openssl
      stdenv.cc.cc
    ];

  shell = pkgs.mkShell {
    buildInputs = [
      pythonWithPkgs
      # pkgs.autoconf
      # pkgs.pkg-config

      # Misc packages needed for compiling python libs
      pkgs.readline
      pkgs.libffi
      pkgs.openssl

      # Necessary because of messing with LD_LIBRARY_PATH
      pkgs.git
      pkgs.openssh
      pkgs.rsync
    ];

    shellHook = ''
      # Allow the use of wheels.
      SOURCE_DATE_EPOCH=$(date +%s)

      # Augment the dynamic linker path
      export "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${lib-path}"

      # Setup the virtual environment if it doesn't already exist.
      VENV=.venv
      if test ! -d $VENV; then
        virtualenv $VENV
      fi

      source ./$VENV/bin/activate
      export PYTHONPATH=`pwd`/$VENV/${myPython.sitePackages}/:$PYTHONPATH

      pip install -r requirements.txt

      python main.py
    '';
  };
in
  shell
