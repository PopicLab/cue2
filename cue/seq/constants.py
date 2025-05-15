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

import cue.utils.types
from collections import namedtuple


SVSignal = cue.utils.types.make_enum("SVSignal",
    ["RD", "RD_LOW", "RD_DIV",# read depth
     "RD_CLIPPED", "RD_CLIPPED_L", "RD_CLIPPED_R", # depth: clipping
     "RD_F", "RD_F_LOW", # depth: fragments
     "LL", "RR", "RL", "LR", # read-pair orientation (short reads)
     "SRORD12", "SRORD21", "SRFLIP", # orientation: split-reads
     "SFORD12", "SFORD21", "SFFLIP", # orientation: fragments
     "CINS", "CDEL", "DIV", # cigar
     "SM", # split-molecule barcodes
     "IN" # fixed target mode
    ], __name__)

SIGNAL_COMBINATIONS = {"_".join(signal.name for signal in signal_combo): signal_combo
                       for signal_combo in
                       [[SVSignal.LL, SVSignal.RR],
                        [SVSignal.SRORD12, SVSignal.CDEL],
                        [SVSignal.SRORD12, SVSignal.SFORD12, SVSignal.CDEL],
                        [SVSignal.SRFLIP, SVSignal.SFFLIP],
                        [SVSignal.SRORD21, SVSignal.SFORD21],
                        [SVSignal.RD_DIV, SVSignal.RD]]}

COLLECTION_SIGNALS = [SVSignal.SM]
SCALAR_SIGNALS = [SVSignal.RD, SVSignal.RD_LOW, SVSignal.RD_CLIPPED_L, SVSignal.RD_CLIPPED_R, SVSignal.RD_CLIPPED,
                  SVSignal.RD_DIV, SVSignal.RD_F, SVSignal.RD_F_LOW, SVSignal.CINS, SVSignal.DIV]

PairOrientation = cue.utils.types.make_enum("PairOrientation",
                                            ["LR", "LL", "RR", "RL"], __name__)
SplitReadOrientation = cue.utils.types.make_enum("SplitReadOrientation",
                                                 ["ORD12", "ORD21", "STRAND_FLIP"], __name__)
SRO_TO_SIGNAL = {SplitReadOrientation.ORD12: SVSignal.SRORD12,
                 SplitReadOrientation.ORD21: SVSignal.SRORD21,
                 SplitReadOrientation.STRAND_FLIP: SVSignal.SRFLIP}

SPO_TO_SIGNAL = {SplitReadOrientation.ORD12: SVSignal.SFORD12,
                 SplitReadOrientation.ORD21: SVSignal.SFORD21,
                 SplitReadOrientation.STRAND_FLIP: SVSignal.SFFLIP}

SARecord = namedtuple('SARecord',
                      ['read_start', 'read_end', 'ref_start', 'ref_end', 'is_reverse', 'mapq'])
SRJunction = namedtuple('SRJunction', ['reference_pos_L', 'reference_pos_R',
                                       'reference_start', 'next_reference_start',
                                       'reference_end', 'next_reference_end',
                                       'read_start', 'next_read_start',
                                       'is_reverse', 'mate_is_reverse', 'mapq', 'mate_mapq'])
CIGAR_OPS = ['M', 'I', 'D', 'N', 'S', 'H', 'P', '=', 'X', 'B']