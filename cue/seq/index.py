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
import cue.seq.io as io
from cue.seq.utils import *
from cue.utils.types import NestedDict
import bisect
from collections import defaultdict
from intervaltree import IntervalTree
import logging
import mappy
import math
import numpy as np
import operator
import os
import os.path
import pickle
import pysam

class CueIndex:
    def __init__(self, chr_name, config):
        self.chr_name = chr_name
        self.config = config
        self.chr_index = io.ChrFAIndex.load(config.fai)
        self.chr = self.chr_index.get_chr_by_name(chr_name)
        self.ref = pysam.Fastafile(config.fa)
        self.scalar_bins = NestedDict(NestedDict(int))
        self.scalar_sparse_bins = NestedDict(NestedDict(int))
        self.collection_bins = NestedDict(NestedDict(set))
        self.adjacencies = NestedDict(NestedDict(NestedDict(int)))
        self.adjacencies_binned = NestedDict(NestedDict(NestedDict(int)))
        self.sparse_counts = {}
        self.adjacency_tree = defaultdict(IntervalTree)
        self.seen_index_reads = set()

    ##################################
    #   Index generation / updates   #
    ##################################

    @staticmethod
    def generate(chr_name, config):
        fname = CueIndex.get_fname(chr_name, config)
        logging.info("Generating %s Cue index: %s" % (chr_name, fname))
        cue_index = CueIndex(chr_name, config)
        cue_index.collect_discordance_signals()
        cue_index.ref = None
        cue_index.store(fname)
        if config.store_adj: cue_index.store_adj()
        return cue_index

    def collect_discordance_signals(self):
        for aln in io.bam_iter(self.config.bam, self.chr_name):
            if (aln.is_unmapped or aln.is_secondary or aln.mapping_quality < self.config.min_mapq or
                "N" in aln.query_alignment_sequence or is_discard(aln, self.config.max_discordance) or
                not is_discordant(aln, self.config.min_discord_len, self.config.min_clipped_len)): continue
            # note: all linked records are processed jointly
            # note: a pair or a split read will be recorded if at least one record passes the MAPQ threshold
            ins_rec = []
            for ref_pos, read_pos, ins_len in get_ins_positions(aln, self.config.min_discord_len):
                self.scalar_sparse_bins[SVSignal.CINS][self.get_bin_id(ref_pos)] += 1
                ins_rec.append((ref_pos, read_pos, ins_len))
            for ref_start, ref_end in get_del_blocks(aln, self.config.min_discord_len, self.config.max_discordance):
                self.record_adjacency([SVSignal.CDEL], [ref_start, ref_end])
            for bin_id in self.get_covered_read_bins(aln):
                self.scalar_bins[SVSignal.RD_DIV][bin_id] += 1
            self.record_frag_signals(aln, ins_rec)
            if is_split(aln):
                if aln.qname in self.seen_index_reads: continue
                self.seen_index_reads.add(aln.qname)
                for j in get_split_read_junctions(aln):
                    ref_start, ref_end = get_split_read_junction_pos(j)
                    if self.config.min_discord_len <= abs(ref_end - ref_start) <= self.config.max_discord_len:
                        self.record_adjacency([SRO_TO_SIGNAL[get_split_read_orientation(j)]],
                                              [ref_start, ref_end])
            elif is_clipped(aln, self.config.min_clipped_len): self.record_clipping(aln)

    def record_adjacency(self, signals, positions):
        for signal in signals:
            p1, p2 = sorted(positions)
            self.adjacencies[p1][p2][signal] += 1
            self.adjacencies_binned[signal][self.get_bin_id(p1)][self.get_bin_id(p2)] += 1


    def record_frag_signals(self, aln, ins_rec):
        ref_seq = self.ref.fetch(self.chr.name, aln.reference_start, aln.reference_end)
        aligner = self.get_aligner(ref_seq, "custom")
        last_hit = None
        for i in range(0, len(aln.query_alignment_sequence) - self.config.frag_len - 1, self.config.frag_len):
            if (hit := next(aligner.map(aln.query_alignment_sequence[i:i + self.config.frag_len]), None)) is None:
                last_hit = None
                continue
            start_bin = self.get_bin_id(hit.r_st + aln.reference_start)
            for bin_id in range(math.ceil((hit.r_en - hit.r_st)/self.config.bin_size)):
                self.scalar_bins[SVSignal.RD_F_LOW][start_bin + bin_id] += 1  # todo: gaps
                if hit.mapq >= self.config.min_mapq: self.scalar_bins[SVSignal.RD_F][start_bin + bin_id] += 1
            # don't record other signals for low MAPQ reads
            if hit.mapq < self.config.min_mapq:
                last_hit = None  # if a read is skipped, don't use the previous fragment for a junction
                continue
            self.collection_bins[SVSignal.SM][start_bin].add(aln.qname)
            if not last_hit:
                last_hit = hit
                continue
            # check for discordant orientation/positions
            j = get_frag_junction(last_hit, i - self.config.frag_len, hit, i, aln.reference_start, self.config.frag_len)
            ref_start, ref_end = sorted(get_split_read_junction_pos(j))
            if self.config.min_discord_len <= abs(ref_end - ref_start) < self.config.max_discord_len:
                orientation = get_split_read_orientation(j)
                ref_start, ref_end = self.process_frag_orientation(aln, orientation, ref_start, ref_end, ins_rec)
                if ref_start is not None: self.record_adjacency([SPO_TO_SIGNAL[orientation]], [ref_start, ref_end])
            last_hit = hit

    def process_frag_orientation(self, aln, orientation, ref_start, ref_end, ins_rec,
                                 max_custom_len=500, size_sim=0.7, seq_sim=0.9):
        if orientation == SplitReadOrientation.ORD12: return None, None
        if orientation == SplitReadOrientation.STRAND_FLIP:
            ref_seq = self.ref.fetch(self.chr.name, ref_start, ref_end)
            aligner = self.get_aligner(aln.query_sequence, "custom" if len(ref_seq) <= max_custom_len else "sr")
            hit = next(aligner.map(ref_seq), None)
            if not hit or hit.strand != -1: return None, None
            span = abs(ref_end - ref_start)
            size_ratio = min(hit.blen, span)/max(hit.blen, span)
            seq_ratio = hit.mlen / hit.blen
            if size_ratio < size_sim or seq_ratio < seq_sim: return None, None
            return ref_start, ref_end
        if orientation == SplitReadOrientation.ORD21:
            span = abs(ref_end - ref_start)
            ins_match = []
            for ref_pos, read_pos, ins_len in ins_rec:
                if ref_pos > ref_end and read_pos < ref_start: continue
                size_ratio = min(ins_len, span)/max(ins_len, span)
                if size_ratio < size_sim: continue
                ins_match.append((size_ratio, ref_pos, read_pos, ins_len))
            if not ins_match: return None, None
            _, ref_pos, read_pos, ins_len = sorted(ins_match, key=lambda x: x[0], reverse=True)[0]
            ref_offset = max(0, ref_pos - ins_len)
            ref_seq = self.ref.fetch(self.chr.name, ref_offset, min(ref_pos + ins_len, self.chr.len))
            aligner = self.get_aligner(ref_seq, "custom" if ins_len <= max_custom_len else "sr")
            mlen = 0
            coords = []
            for hit in aligner.map(aln.seq[read_pos:read_pos+ins_len]):
                if not hit.is_primary: continue
                mlen += hit.mlen
                coords.extend([hit.r_st, hit.r_en])
            if mlen/ins_len < seq_sim: return None, None
            return ref_offset + min(coords), ref_offset + max(coords)

    def record_clipping(self, aln, seq_sim=0.9):
        ref_seq = self.ref.fetch(self.chr.name, aln.reference_start, aln.reference_end)
        aligner = self.get_aligner(ref_seq)
        clipped_seq = []
        if aln.cigartuples[0][0] == 4 and aln.cigartuples[0][1] >= self.config.min_clipped_len:
            clipped_seq.append((0, aln.query_sequence[:aln.cigartuples[0][1]]))
        if aln.cigartuples[-1][0] == 4 and aln.cigartuples[-1][1] >= self.config.min_clipped_len:
            clipped_seq.append((-1, aln.query_sequence[-aln.cigartuples[-1][1]:]))
        for rend, seq in clipped_seq:
            hit = next(aligner.map(seq), None)
            if not hit or hit.mlen / len(seq) < seq_sim: continue
            if aln.is_reverse != (hit.strand == "-1"): continue
            if rend == 0:
                ref_start = aln.reference_start + hit.r_en if hit.strand != "-1" else aln.reference_start + hit.r_st
                ref_end = aln.reference_start if not aln.is_reverse else aln.reference_end
            else:
                ref_start = aln.reference_end if not aln.is_reverse else aln.reference_start
                ref_end = aln.reference_start + hit.r_st if hit.strand != "-1" else aln.reference_start + hit.r_en
            self.record_adjacency([SVSignal.SFORD21], [ref_start, ref_end])

    def record_extra_signals(self, intervals):
        if not intervals: return
        current_interval_idx = 0
        for aln in io.bam_iter(self.config.bam, self.chr.name):
            if (aln.is_unmapped or aln.is_secondary or ("N" in aln.query_alignment_sequence) or
                    is_discard(aln, self.config.max_discordance)): continue
            if aln.reference_end < intervals[current_interval_idx][0]: continue
            while (current_interval_idx < len(intervals) and
                   intervals[current_interval_idx][1] < aln.reference_start):
                current_interval_idx += 1
            if current_interval_idx >= len(intervals): return
            # ----- single-record signals: read depth and clipping
            covered_bins = self.get_covered_read_bins(aln)
            for bin_id in covered_bins:
                self.scalar_bins[SVSignal.RD_LOW][bin_id] += 1
                if aln.mapping_quality >= self.config.min_mapq:
                    self.scalar_bins[SVSignal.RD][bin_id] += 1
            if aln.mapping_quality < self.config.high_mapq: continue
            if is_clipped_end(aln, self.config.min_clipped_len, "left"):
                self.scalar_sparse_bins[SVSignal.RD_CLIPPED_L][self.get_bin_id(aln.reference_start)] += 1
            if is_clipped_end(aln, self.config.min_clipped_len, "right"):
                self.scalar_sparse_bins[SVSignal.RD_CLIPPED_R][self.get_bin_id(aln.reference_end)] += 1

    ###################
    #   Index utils   #
    ###################

    def get_aligner(self, seq, preset="sr"):
        if preset=="custom": return mappy.Aligner(seq=seq, preset="sr", best_n=1, min_cnt=1, k=15, w=5,
                                                  min_dp_score=10, min_chain_score=1)
        return mappy.Aligner(seq=seq, preset=preset, best_n=1)

    def get_bin_id(self, pos):
        return pos // self.config.bin_size

    def get_covered_read_bins(self, read):
        bins = set()
        if not has_cigar_discordance(read, self.config.min_discord_len):
            start_bin = self.get_bin_id(read.reference_start)
            reference_span = read.reference_end - read.reference_start
            return [start_bin + i for i in range(math.ceil(reference_span / self.config.bin_size))]
        for block_start, block_end in read.get_blocks():
            start_bin = self.get_bin_id(block_start)
            reference_span = block_end - block_start
            bins.update([start_bin + i for i in range(math.ceil(reference_span / self.config.bin_size))])
        return bins

    def build_lookup_db(self):
        for signal in self.adjacencies_binned.keys():
            for i in self.adjacencies_binned[signal].keys():
                for j in self.adjacencies_binned[signal][i].keys():
                    if i == j: continue
                    self.adjacency_tree[signal].addi(i, j, self.adjacencies_binned[signal][i][j])
        for signal in self.scalar_sparse_bins:
            self.sparse_counts[signal] = np.transpose(np.array([[k, self.scalar_sparse_bins[signal][k]]
                                           for k in sorted(self.scalar_sparse_bins[signal].keys())]))

    def initialize_grid(self, interval_a, interval_b):
        start_bin_id_a = self.get_bin_id(interval_a.start)
        end_bin_id_a = self.get_bin_id(interval_a.end)
        n_bins_a = end_bin_id_a - start_bin_id_a
        start_bin_id_b = self.get_bin_id(interval_b.start)
        end_bin_id_b = self.get_bin_id(interval_b.end)
        n_bins_b = end_bin_id_b - start_bin_id_b
        return np.zeros((n_bins_a, n_bins_b)), start_bin_id_a, start_bin_id_b

    ########################
    #   Index lookup ops   #
    ########################

    def intersect(self, signals, interval_a, interval_b):
        if len(signals) == 1 and signals[0] in COLLECTION_SIGNALS:
            return self.intersect_collection(signals[0], interval_a, interval_b)
        non_zero = False
        counts, start_bin_id_a, start_bin_id_b = self.initialize_grid(interval_a, interval_b)
        for signal in signals:
            interval_a_overlap = self.adjacency_tree[signal].overlap(start_bin_id_a, start_bin_id_a + counts.shape[0])
            interval_b_overlap = self.adjacency_tree[signal].overlap(start_bin_id_b, start_bin_id_b + counts.shape[0])
            for adj in interval_a_overlap.intersection(interval_b_overlap):
                bin_a = adj.begin
                bin_b = adj.end
                if bin_a == bin_b: continue
                if bin_a < start_bin_id_a: continue
                if bin_b >= start_bin_id_b + counts.shape[0]: continue
                non_zero = True
                i = bin_a - start_bin_id_a
                j = bin_b - start_bin_id_b
                counts[i][j] += adj.data
                if bin_a >= start_bin_id_b and bin_b < start_bin_id_a + counts.shape[0]:
                    counts[bin_b - start_bin_id_a][bin_a - start_bin_id_b] += adj.data
        return counts, non_zero

    def intersect_collection(self, signal, interval_a, interval_b):
        non_zero = False
        counts, start_bin_id_a, start_bin_id_b = self.initialize_grid(interval_a, interval_b)
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                bin_a = i + start_bin_id_a
                bin_b = j + start_bin_id_b
                if not self.collection_bins[signal][bin_a] or not self.collection_bins[signal][bin_b]: continue
                if bin_a != bin_b:
                    counts[i][j] = len(self.collection_bins[signal][bin_a].intersection(self.collection_bins[signal][bin_b]))
                    if counts[i][j] != 0:
                        non_zero = True
        return counts, non_zero

    def scalar_apply(self, signal, interval_a, interval_b, op=operator.sub):
        non_zero = False
        counts, start_bin_id_a, start_bin_id_b = self.initialize_grid(interval_a, interval_b)
        if signal not in self.sparse_counts:
            for i in range(counts.shape[0]):
                for j in range(counts.shape[1]):
                    counts[i][j] = op(self.scalar_bins[signal][i + start_bin_id_a],
                                      self.scalar_bins[signal][j + start_bin_id_b])
                    if not non_zero and counts[i][j] != 0:
                        non_zero = True
        else:
            lower_bound_a = bisect.bisect_left(self.sparse_counts[signal][0,:], start_bin_id_a)
            upper_bound_a = bisect.bisect_right(self.sparse_counts[signal][0,:],
                                                start_bin_id_a + counts.shape[0] - 1, lo=lower_bound_a)
            lower_bound_b = bisect.bisect_left(self.sparse_counts[signal][0,:], start_bin_id_b)
            upper_bound_b = bisect.bisect_right(self.sparse_counts[signal][0,:],
                                                start_bin_id_b + counts.shape[0] - 1, lo=lower_bound_b)
            bin2id_a = {self.sparse_counts[signal][0][bin_a_idx]: bin_a_idx for bin_a_idx in range(lower_bound_a, upper_bound_a)}
            bin2id_b = {self.sparse_counts[signal][0][bin_b_idx]: bin_b_idx for bin_b_idx in range(lower_bound_b, upper_bound_b)}
            for bin_a, bin_a_idx in bin2id_a.items():
                i = self.sparse_counts[signal][0][bin_a_idx] - start_bin_id_a
                for j in range(counts.shape[1]):
                    bin_b = j + start_bin_id_b
                    if bin_b in bin2id_b:
                        counts[i][j] = op(self.sparse_counts[signal][1][bin_a_idx],
                                          self.sparse_counts[signal][1][bin2id_b[bin_b]])
                    else: counts[i][j] = op(self.sparse_counts[signal][1][bin_a_idx], 0)
            for bin_b, bin_b_idx in bin2id_b.items():
                j = self.sparse_counts[signal][0, bin_b_idx] - start_bin_id_b
                for i in range(counts.shape[0]):
                    bin_a = i + start_bin_id_a
                    if bin_a in bin2id_a: continue  # already computed for these bins
                    counts[i][j] = op(0, self.sparse_counts[signal][1][bin_b_idx])
        return counts, non_zero

    #################
    #   Index IO    #
    #################

    @staticmethod
    def get_fname(chr_name, config):
        return "%s/%s.%d.ci" % (config.index_dir, chr_name, config.bin_size)

    def store(self, fname):
        logging.info("Storing Cue index: %s" % fname)
        with open(fname, 'wb') as fp:
            pickle.dump(self, fp)

    @staticmethod
    def load(fname):
        logging.info("Loading Cue index: %s" % fname)
        with open(fname, 'rb') as fp:
            return pickle.load(fp)

    @staticmethod
    def generate_or_load(chr_name, config):
        fname = CueIndex.get_fname(chr_name, config)
        if os.path.isfile(fname): return CueIndex.load(fname)
        return CueIndex.generate(chr_name, config)

    def store_adj(self):
        adj_fname = "%s/%s.adj.vcf" % (self.config.index_dir, self.chr_name,)
        header = pysam.VariantHeader()
        header.add_line('##INFO=<ID=ADJTYPE,Number=1,Type=String,Description="Adjacency type">')
        header.add_line('##INFO=<ID=COUNT,Number=1,Type=Integer,Description="Adjacency count">')
        for ctg in self.chr_index.contigs(): header.add_line("##contig=<ID=%s>\n" % ctg.name)
        adj_vcf = pysam.VariantFile(adj_fname, 'w', header=header)
        for p1 in self.adjacencies:
            for p2 in self.adjacencies[p1]:
                for signal in self.adjacencies[p1][p2]:
                    adj_vcf.write(adj_vcf.new_record(contig=self.chr_name, start=p1, stop=p2,
                        info={"ADJTYPE": signal.name, "COUNT": self.adjacencies[p1][p2][signal]},
                                                     alleles=('N', '<NA>'), id="ADJ",
                                                     qual=None, filter=None, samples=None))
        adj_vcf.close()
        logging.info("Wrote %s file" % adj_fname)
