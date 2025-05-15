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

from cue.seq.constants import *
from bitarray import bitarray
import random
import numpy as np


def partition_chrs(chr_names, n_chunks):
    random.shuffle(chr_names)
    return np.array_split(np.array(list(chr_names)), n_chunks)

def seq_to_num(seq):
    base2num = {'A': bitarray('00'), 'C': bitarray('01'), 'G': bitarray('10'), 'T': bitarray('11')}
    seq_bits = bitarray()
    seq_bits.encode(base2num, seq[:-2])
    return int(seq_bits.to01(), 2)

# ------- CIGAR utils ----------
def get_ins_positions(read, min_ins_len):
    insertions = []
    ref_pos = read.reference_start
    read_pos = 0
    for op, len in read.cigartuples:
        if CIGAR_OPS[op] == 'I':
            if len >= min_ins_len: insertions.append((ref_pos, read_pos, len))
            read_pos += len
        if CIGAR_OPS[op] in ["M", "D", "N", "=", "X"]: ref_pos += len
        if CIGAR_OPS[op] in ["S", "M", "X", "="]: read_pos += len
    return insertions

def get_del_blocks(read, min_del_len, max_discordance, min_filt_len=100):
    del_blocks = []
    block_discordance = []
    ref_pos = read.reference_start
    n_mismatch = 0
    n_match = 0
    for op, op_len in read.cigartuples:
        if CIGAR_OPS[op] == 'D' and op_len >= min_del_len:
            del_blocks.append((ref_pos, ref_pos + op_len))
            block_discordance.append((n_mismatch, n_match))
            n_mismatch = 0
            n_match = 0
        if CIGAR_OPS[op] in ["M", "D", "N", "=", "X"]: ref_pos += op_len
        if CIGAR_OPS[op] in ["X", "D", "I"]: n_mismatch += 1
        elif CIGAR_OPS[op] in ["M", "="]: n_match += op_len
    block_discordance.append((n_mismatch, n_match))
    adj = []
    for i, del_adj in enumerate(del_blocks):
        if i > 0:
            adj.append(del_adj)
            continue
        n_mismatch1, n_match1 = block_discordance[i]
        n_mismatch2, n_match2 = block_discordance[i+1]
        block_discordance1 = n_mismatch1 / (n_mismatch1 + n_match1)
        block_discordance2 = n_mismatch2 / (n_mismatch2 + n_match2)
        if ((n_match1 + n_mismatch1) < min_filt_len) or ((n_match2 + n_mismatch2) < min_filt_len):
            adj.append(del_adj)
            continue
        if block_discordance1 > max_discordance or block_discordance2 > max_discordance: continue
        adj.append(del_adj)
    return adj

def get_div_blocks(read):
    div_blocks = []
    ref_pos = read.reference_start
    for op, len in read.cigartuples:
        if CIGAR_OPS[op] == 'X':
            div_blocks.append((ref_pos, ref_pos + len))
        if CIGAR_OPS[op] in ["M", "D", "N", "=", "X"]:
            ref_pos += len
    return div_blocks

def is_divergent_cigar(read, min_discord_len, min_match_len=25):
    div_block = 0
    for op, len in read.cigartuples:
        if CIGAR_OPS[op] == 'I' and len >= min_discord_len: return True
        if CIGAR_OPS[op] == 'D' and len >= min_discord_len: return True
        if CIGAR_OPS[op] in ["X", "D", "I"]:
            div_block += len
            if div_block >= min_discord_len: return True
        elif CIGAR_OPS[op] in ["M", "="] and len > min_match_len:  # breaks the div block
            div_block = 0
    return False

def has_cigar_discordance(read, min_discord_len):
    for op, len in read.cigartuples:
        if CIGAR_OPS[op] == 'I' and len >= min_discord_len: return True
        if CIGAR_OPS[op] == 'D' and len >= min_discord_len: return True
    return False

def get_reference_span(cigar):
    ref_span = 0
    cigar_idx = 0
    op_len_str = ""
    while cigar_idx < len(cigar):
        if not cigar[cigar_idx].isdigit():
            if cigar[cigar_idx] in ["M", "D", "N", "=", "X"]:
                ref_span += int(op_len_str)
            op_len_str = ""
        else:
            op_len_str += cigar[cigar_idx]
        cigar_idx += 1
    return ref_span

def get_read_start(cigar, strand):
    op_len_str = ""
    if strand == "+":
        cigar_idx = 0
        while cigar[cigar_idx].isdigit():
            op_len_str += cigar[cigar_idx]
            cigar_idx += 1
        if cigar[cigar_idx] in ["H", "S"]:
            return int(op_len_str)
    elif cigar[-1] in ["H", "S"]:
        cigar_idx = -2
        while cigar[cigar_idx].isdigit():
            op_len_str += cigar[cigar_idx]
            cigar_idx -= 1
        return int(op_len_str[::-1])
    return 0

def get_read_len(cigar):
    rlen = 0
    cigar_idx = 0
    op_len_str = ""
    while cigar_idx < len(cigar):
        if cigar[cigar_idx].isdigit():
            op_len_str += cigar[cigar_idx]
        else:
            if cigar[cigar_idx] in ["M", "I", "=", "X"]:
                rlen += int(op_len_str)
            op_len_str = ""
        cigar_idx += 1
    return rlen

def is_clipped_end(read, min_clipped_len, end):
    # soft (op 4) or hard clipped (op 5)
    # end: "left" or "right"
    cigar_end = 0 if end == "left" else -1
    return read.cigartuples[cigar_end][0] in [4, 5] and read.cigartuples[cigar_end][1] > min_clipped_len

def is_clipped(read, min_clipped_len):
    return is_clipped_end(read, min_clipped_len, "left") or is_clipped_end(read, min_clipped_len, "right")


# ------- Alignment utils ----------
def is_split(read):
    # todo: adapt for inter-chromosome functionality
    if read.has_tag("SA"):
        for sa_tag in read.get_tag('SA').rstrip(";").split(';'):
            if read.reference_name == sa_tag.split(',')[0]: return True
    return False

def is_discordant(aln, min_discord_len, min_clip_len):
    return is_split(aln) or is_divergent_cigar(aln, min_discord_len) or is_clipped(aln, min_clip_len)

def is_discard(aln, max_discordance):
    if aln.has_tag("SA"):
        for sa_tag in aln.get_tag('SA').rstrip(";").split(';'):
            if aln.reference_name != sa_tag.split(',')[0]: return True
    n_match = 0
    n_mismatch = 0
    for op, op_len in aln.cigartuples:
        if CIGAR_OPS[op] in ["X", "D", "I"]: n_mismatch += 1
        elif CIGAR_OPS[op] in ["M", "="]: n_match += op_len
    return n_mismatch / (n_mismatch + n_match) >= max_discordance

def is_paired_and_discordant(read, min_discord_distance):
    # read pair must be mapped to the same chromosome
    # read pair distance must be within the configured range
    orientation = get_pair_orientation(read)
    return (read.is_paired and not read.mate_is_unmapped and read.next_reference_name == read.reference_name and
            (orientation != PairOrientation.LR or
             abs(read.reference_start - read.next_reference_start) >= min_discord_distance))

def is_singleton(read):
    return read.mate_is_unmapped or read.next_reference_name != read.reference_name

def get_mate_discordant_pair_end(read, chr_len):
    # approximate the end position for the mate
    return min(read.next_reference_start + len(read.seq), chr_len - 1)

# ------- Junction/orientation utils ----------
def get_pair_orientation(read, rl_dist_thr=5):
    if read.is_reverse == read.mate_is_reverse:
        return PairOrientation.RR if read.is_reverse else PairOrientation.LL
    if ((read.reference_start + rl_dist_thr) < read.next_reference_start and read.is_reverse) or \
            (read.reference_start > (read.next_reference_start + rl_dist_thr) and not read.is_reverse):
        return PairOrientation.RL
    return PairOrientation.LR

def get_split_read_orientation(junction, overlap_thr=50):
    if junction.is_reverse != junction.mate_is_reverse: return SplitReadOrientation.STRAND_FLIP
    overlap = junction.reference_pos_L - junction.reference_pos_R
    if (not junction.is_reverse and overlap > overlap_thr) or (junction.is_reverse and overlap < -overlap_thr):
        return SplitReadOrientation.ORD21
    return SplitReadOrientation.ORD12

def get_discordant_pair_pos(read, chr_len):
    orientation = get_pair_orientation(read)
    mate_end = get_mate_discordant_pair_end(read, chr_len)
    if orientation == PairOrientation.LR: return read.reference_end, read.next_reference_start
    if orientation == PairOrientation.RL: return read.reference_start, mate_end
    if orientation == PairOrientation.LL: return read.reference_end, mate_end
    if orientation == PairOrientation.RR: return read.reference_start, read.next_reference_start

def get_split_read_junction_pos(junction):
    return junction.reference_pos_L, junction.reference_pos_R

def get_split_read_junctions(read, max_segment_missing=15):
    # order split alignments according to their position in the read
    # junctions are created between adjacent positions in the read
    pos2sa = {}
    read_start = get_read_start(read.cigarstring, "-" if read.is_reverse else "+")
    read_end = read_start + get_read_len(read.cigarstring)
    pos2sa[read_start] = SARecord(read_start, read_end, read.reference_start, read.reference_end, read.is_reverse,
                                  read.mapping_quality)
    for sa_tag in read.get_tag('SA').rstrip(";").split(';'):
        entries = sa_tag.split(',')
        if read.reference_name == entries[0]:
            ref_start = int(entries[1])
            strand, cigar, mapq, _ = entries[2:]
            if int(mapq) < 1: continue
            ref_end = ref_start + get_reference_span(cigar)
            read_start = get_read_start(cigar, strand)
            read_end = read_start + get_read_len(cigar)
            pos2sa[read_start] = SARecord(read_start, read_end, ref_start, ref_end, True if strand == "-" else False, mapq)
    ordered_sa = sorted(pos2sa.keys())
    junctions = []
    for i in range(1, len(ordered_sa)):
        sa_first = pos2sa[ordered_sa[i-1]]
        sa_second = pos2sa[ordered_sa[i]]
        # no junction when a read segment is dropped by the aligner
        if sa_second.read_start > sa_first.read_end + max_segment_missing: continue
        junctions.append(SRJunction(sa_first.ref_end if not sa_first.is_reverse else sa_first.ref_start,
                                    sa_second.ref_start if not sa_second.is_reverse else sa_second.ref_end,
                                    sa_first.ref_start, sa_second.ref_start,
                                    sa_first.ref_end, sa_second.ref_end,
                                    sa_first.read_start, sa_second.read_start,
                                    sa_first.is_reverse, sa_second.is_reverse,
                                    sa_first.mapq, sa_second.mapq))
    return junctions

def get_frag_junction(aln1, read_offset1, aln2, read_offset2, ref_offset, read_len):
    # already sorted by position in the read
    ref_start1 = ref_offset + aln1.r_st
    ref_end1 = ref_offset + aln1.r_en
    ref_start2 = ref_offset + aln2.r_st
    ref_end2 = ref_offset + aln2.r_en
    is_reverse1 = aln1.strand == -1
    is_reverse2 = aln2.strand == -1
    j1_shift = j2_shift = 0
    if not is_reverse1:
        j1 = ref_end1
        if aln1.q_en < read_len:
            j2_shift = read_len - aln1.q_en
        if aln2.q_st > 0:
            j1_shift = aln2.q_st
    else:
        j1 = ref_start1
        if aln1.q_en < read_len:
            j2_shift = -(read_len - aln1.q_en)
        if aln2.q_st > 0:
            j1_shift = -aln2.q_st
    if not is_reverse2: j2 = ref_start2
    else: j2 = ref_end2
    j1 = j1 + j1_shift
    j2 = j2 + j2_shift
    return SRJunction(j1, j2, ref_start1, ref_start2, ref_end1, ref_end2, read_offset1, read_offset2,
                      is_reverse1, is_reverse2, aln1.mapq, aln2.mapq)
