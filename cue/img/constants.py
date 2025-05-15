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

from cue.seq.constants import SVSignal, SIGNAL_COMBINATIONS
import cue.utils.types


# -------- Image channels --------
Channel = cue.utils.types.make_enum("Channel",
                                    [signal_name for signal_name in SVSignal.__members__] +
                                    [signal_combo_name for signal_combo_name in SIGNAL_COMBINATIONS], __name__)
CHANNEL_TO_SIGNAL = {Channel[signal]: SVSignal[signal] for signal in SVSignal.__members__}
CHANNEL_TO_SIGNAL_COMBO = {Channel[channel_name]: SIGNAL_COMBINATIONS[channel_name]
                           for channel_name in Channel.__members__ if channel_name not in SVSignal.__members__}

# -------- Image channel sets (e.g. channels corresponding to a specific platform/regime) --------
ChannelSet = cue.utils.types.make_enum("SvChannelSet",
                                       ["SHORT", "LONG"], __name__)
CHANNELS_BY_TYPE = {
    ChannelSet.SHORT: [Channel.RD, Channel.RD_LOW,
                       Channel.LR, Channel.RL, Channel.LL_RR,
                       Channel.SRORD12, Channel.SRORD21, Channel.SRFLIP],
    ChannelSet.LONG:  [Channel.RD, Channel.RD_CLIPPED_L, Channel.RD_CLIPPED_R,
                       Channel.SRORD12_CDEL, Channel.SRORD21, Channel.SRFLIP,
                       Channel.RD_F_LOW, Channel.SM, Channel.SFORD21, Channel.SFFLIP,
                       Channel.CINS, Channel.RD_DIV_RD],
}

CHANNEL_SET_TO_CHANNEL_IDX = {
    ChannelSet.SHORT: {ChannelSet.SHORT: range(len(CHANNELS_BY_TYPE[ChannelSet.SHORT]))},
    ChannelSet.LONG: {ChannelSet.LONG: range(len(CHANNELS_BY_TYPE[ChannelSet.LONG]))},
}

# -------- Image annotations --------
TargetType = cue.utils.types.make_enum("TargetType",
                                       ["boxes", "keypoints", "labels", "classes", "image_id", "area",
                                        "heatmaps", "weight", "scores", "gloc", "dataset_id"], __name__)

# -------- Image keypoints --------
KP_VISIBLE = 1
KP_FILTERED = -1