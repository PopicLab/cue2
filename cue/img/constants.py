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


from enum import Enum
from collections import namedtuple
from cue.seq.constants import *


# Image channels
class Channel(str, Enum):
    RD = "RD"
    RD_LOW = "RD_LOW"
    RD_CLIPPED = "RD_CLIPPED"
    RD_CLIPPED_L = "RD_CLIPPED_L"
    RD_CLIPPED_R = "RD_CLIPPED_R"
    RP = "RP"
    RD_SP = "RD_SP"
    RD_SP_LOW = "RD_SP_LOW"
    RD_DIV = "RD_DIV"
    RD_MAX = "RD_MAX"
    RD_DIV_MAX = "RD_DIV_MAX"
    SR = "SR"
    SR_RP = "SR_RP"
    LR = "LR"
    LLRR = "LLRR"
    RL = "RL"
    RP_VS_RD = "RP_VS_RD"
    LLRR_VS_LR = "LLRR_VS_LR"
    SM = "SM"
    SRC = "SRC"
    SRS = "SRS"
    SRI = "SRI"
    SPI = "SPI"
    SPS = "SPS"
    SPC = "SPC"
    INS = "INS"
    DEL = "DEL"
    DIV = "DIV"
    SRC_DEL = "SRC_DEL"
    SRC_SPC_DEL = "SRC_SPC_DEL"
    SRI_SPI = "SRI_SPI"
    SRS_SPS = "SRI_SPI"
    DIV_VS_RD = "DIV_VS_RD"


CHANNEL_TO_SIGNAL = {Channel.RD: SVSignals.RD,
                     Channel.RD_LOW: SVSignals.RD_LOW,
                     Channel.RD_CLIPPED: SVSignals.RD_CLIPPED,
                     Channel.RD_CLIPPED_L: SVSignals.RD_CLIPPED_L,
                     Channel.RD_CLIPPED_R: SVSignals.RD_CLIPPED_R,
                     Channel.RD_SP: SVSignals.RD_SP,
                     Channel.RD_SP_LOW: SVSignals.RD_SP_LOW,
                     Channel.RD_DIV: SVSignals.RD_DIV,
                     Channel.LR: SVSignals.LR,
                     Channel.RL: SVSignals.RL,
                     Channel.SR: SVSignals.SR,
                     Channel.SRC: SVSignals.SRC,
                     Channel.SRS: SVSignals.SRS,
                     Channel.SRI: SVSignals.SRI,
                     Channel.SPI: SVSignals.SPI,
                     Channel.SPS: SVSignals.SPS,
                     Channel.SPC: SVSignals.SPC,
                     Channel.SM: SVSignals.SM,
                     Channel.INS: SVSignals.INS,
                     Channel.DEL: SVSignals.DEL,
                     Channel.DIV: SVSignals.DIV,
                     }

CHANNEL_TO_SIGNAL_COMBINATIONS = {Channel.SR_RP: [SVSignals.SR, SVSignals.RP],
                                  Channel.LLRR: [SVSignals.LL, SVSignals.RR],
                                  Channel.SRC_DEL: [SVSignals.SRC, SVSignals.DEL],
                                  Channel.SRC_SPC_DEL: [SVSignals.SRC, SVSignals.SPC, SVSignals.DEL],
                                  Channel.SRI_SPI: [SVSignals.SRI, SVSignals.SPI],
                                  Channel.SRS_SPS: [SVSignals.SRS, SVSignals.SPS]
                                  }

CHANNELS_TO_CACHE = [Channel.SR_RP, Channel.LLRR]

SV_CHANNEL_SET = Enum("SV_CHANNEL_SET", 'SHORT '
                                        'LONG '
                                        'LINKED '
                                        'EMPTY ')

SV_CHANNELS_BY_TYPE = {
    SV_CHANNEL_SET.SHORT: [Channel.RD, Channel.RD_LOW, Channel.RP_VS_RD,
                           Channel.LR, Channel.RL, Channel.LLRR,
                           Channel.SRC, Channel.SRS, Channel.SRI],
    SV_CHANNEL_SET.LINKED: [Channel.SM, Channel.RD_LOW, Channel.SR_RP, Channel.LLRR, Channel.RL],
    SV_CHANNEL_SET.LONG:  [Channel.RD, Channel.RD_CLIPPED_L, Channel.RD_CLIPPED_R,
                           Channel.SRC_DEL, Channel.SRS, Channel.SRI,
                           Channel.RD_SP_LOW, Channel.SM, Channel.SPS, Channel.SPI,
                           Channel.INS, Channel.DIV_VS_RD],
    SV_CHANNEL_SET.EMPTY: []
}

SV_SIGNAL_SET_CHANNEL_IDX = {
    SV_CHANNEL_SET.SHORT: {SV_CHANNEL_SET.SHORT: range(len(SV_CHANNELS_BY_TYPE[SV_CHANNEL_SET.SHORT]))},
    SV_CHANNEL_SET.LONG: {SV_CHANNEL_SET.LONG: range(len(SV_CHANNELS_BY_TYPE[SV_CHANNEL_SET.LONG]))},
    SV_CHANNEL_SET.LINKED: {SV_CHANNEL_SET.LINKED: range(len(SV_CHANNELS_BY_TYPE[SV_CHANNEL_SET.LINKED]))}
}

# Image classes
SV_CLASS_SET = Enum("SV_CLASS_SET", 'BASIC4 '
                                    'BASIC4ZYG '
                                    'BASIC5 '
                                    'BASIC5ZYG '
                                    'BINARY')

SV_CLASSES = {SV_CLASS_SET.BASIC4: ["NEG", "DEL", "INV", "DUP"],
              SV_CLASS_SET.BASIC5: ["NEG", "DEL", "INV", "DUP", "INVDUP"],
              SV_CLASS_SET.BASIC4ZYG: ["NEG", "DEL-HOM", "INV-HOM", "DUP-HOM", "DEL-HET", "INV-HET", "DUP-HET"],
              SV_CLASS_SET.BASIC5ZYG: ["NEG", "DEL-HOM", "INV-HOM", "DUP-HOM", "DEL-HET", "INV-HET", "DUP-HET",
                                       "INVDUP-HOM", "INVDUP-HET"],
              SV_CLASS_SET.BINARY: ["NEG", "POS"]}

CLASS_BACKGROUND = "NEG"
CLASS_SV = "POS"

SV_ZYGOSITY_SETS = {SV_CLASS_SET.BASIC4ZYG, SV_CLASS_SET.BASIC5ZYG}

SV_LABELS = {"NEG": 0, "POS": 1, "DEL": 1, "INV": 2, "DUP": 3,
             "DEL-HOM": 1, "INV-HOM": 2, "DUP-HOM": 3,
             "DEL-HET": 4, "INV-HET": 5, "DUP-HET": 6,
             "IDUP": 4, "IDUP-HOM": 7, "IDUP-HET": 8,
             "INVDUP": 4, "INVDUP-HOM": 7, "INVDUP-HET": 8,
             "TRA": 5, "TRA-HOM": 9, "TRA-HET": 10}

SV_LABELS_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3, 7: 4, 8: 4}
SV_CLASS_MAP = {SV_CLASS_SET.BASIC5ZYG: SV_CLASS_SET.BASIC5,
                SV_CLASS_SET.BASIC5: SV_CLASS_SET.BASIC5,
                SV_CLASS_SET.BASIC4ZYG: SV_CLASS_SET.BASIC4,
                SV_CLASS_SET.BASIC4: SV_CLASS_SET.BASIC4}

LABEL_BACKGROUND = 0
LABEL_SV = 1
LABEL_LANDMARK_DEFAULT = 0
KP_VISIBLE = 1
KP_FILTERED = -1


class TargetType(str, Enum):
    boxes = "boxes"
    keypoints = "keypoints"
    labels = "labels"
    classes = "classes"
    image_id = "image_id"
    area = "area"
    heatmaps = "heatmaps"
    weight = "weight"
    scores = "scores"
    gloc = "gloc"
    dataset_id = "dataset_id"


class ZYGOSITY(str, Enum):
    HET = "HET"
    HOM = "HOM"
    UNK = "UNK"
    HOMREF = "HOMREF"


ZYGOSITY_ENCODING_SIM = {"homAB": ZYGOSITY.HOM, "hetA": ZYGOSITY.HET, "hetB": ZYGOSITY.HET, "UNK": ZYGOSITY.UNK}
ZYGOSITY_ENCODING = {(0, 1): ZYGOSITY.HET, (1, 1): ZYGOSITY.HOM, (1, 0): ZYGOSITY.HET, (0, 0): ZYGOSITY.HOMREF,
                     (None, None): ZYGOSITY.UNK}
ZYGOSITY_ENCODING_BED = {"0/1": ZYGOSITY.HET, "1/1": ZYGOSITY.HOM, "1/0": ZYGOSITY.HET, "./.": ZYGOSITY.UNK}
ZYGOSITY_GT_BED = {ZYGOSITY.HOM: "1/1", ZYGOSITY.HET: "0/1", ZYGOSITY.UNK: "./."}
ZYGOSITY_GT_VCF = {ZYGOSITY.HOMREF: (0, 0), ZYGOSITY.HOM: (1, 1), ZYGOSITY.HET: (0, 1), ZYGOSITY.UNK: (None, None)}
