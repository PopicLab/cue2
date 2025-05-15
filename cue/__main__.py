# Copyright (c) 2023-2025 Victoria Popic
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU Affero General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU Affero General Public License for more details.

#You should have received a copy of the GNU Affero General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.

import cue
import argparse
import importlib


def main():
    parser = argparse.ArgumentParser(prog="Cue (%s)" % cue.__version__,
                                     usage="cue <command> --config <file>",
                                     description='Deep learning framework for structural variant discovery')
    parser.add_argument("command", help="Cue command to run (call, train, generate)")
    parser.add_argument('--config', required=True, help='Cue YAML config file')
    parser.add_argument("--version", action="version", version=cue.__version__)
    args = parser.parse_args()
    if not args.command in ["call", "train", "generate"]:
        print("[ERROR] Unknown command: '%s'" % args.command)
        parser.print_usage()
        exit(1)
    cue_module = importlib.import_module("cue.cli." + args.command)
    cue_module.main(args.config)

if __name__ == "__main__":
    main()

