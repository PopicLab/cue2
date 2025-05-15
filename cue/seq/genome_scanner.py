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


from cue.seq.constants import SVSignal
from cue.seq.intervals import GenomeInterval, GenomeIntervalPair
from cue.seq.sv import SVContainer
from cue.utils.types import NestedDict
from collections import Counter, defaultdict
import logging
from tqdm import tqdm

class GenomeScanner:
    def __init__(self, cue_index, interval_size, step_size):
        self.aln_index = cue_index
        self.chr = cue_index.chr
        self.interval_size = interval_size
        self.step_size = step_size

    def log_intervals(self, x, y):
        logging.debug("Interval pair: %s x=%d y=%d" % (self.chr.name, x, y))


class TargetIntervalScanner(GenomeScanner):
    def __init__(self, config, aln_index):
        super().__init__(aln_index, config.interval_size, config.step_size)
        self.config = config
        self.interval_pair_support = NestedDict(NestedDict(int))
        self.interval_pair_support_adj = NestedDict(list)
        self.interval_pairs = []
        self.intervals = []
        self.steps_per_interval = self.config.interval_size // self.config.step_size
        if self.config.fixed_targets:
            logging.info("Generating fixed target intervals")
            for sv in SVContainer(self.config.fixed_targets, aln_index.chr_index):
                if sv.chr_name != self.chr.name: continue
                self.record_interval_pair_support(sv.start, sv.end, SVSignal.IN, 1000)
        else:
            self.process_adjacencies()
        self.select_interval_pairs()
        if self.interval_pairs: self.record_intervals()
        self.aln_index.record_extra_signals(self.intervals)
        self.aln_index.build_lookup_db()

    def record_interval_pair_support(self, p1, p2, signal_type, count):
        adj_span = abs(p1 - p2)
        if adj_span < self.config.min_sv_len or adj_span > self.config.max_sv_len: return
        intv_ids_p1 = self.get_interval_ids(p1)
        intv_ids_p2 = self.get_interval_ids(p2)
        intv_pairs = set()
        for i in sorted(set(intv_ids_p1).intersection(intv_ids_p2)): intv_pairs.add((i, i))
        j_mid = intv_ids_p2[len(intv_ids_p2) // 2]
        i_mid = intv_ids_p1[len(intv_ids_p1) // 2]
        intv_pairs.add(tuple(sorted((i_mid, j_mid))))
        if self.config.high_redundancy or len(intv_pairs) < 2:
            for i in intv_ids_p1: intv_pairs.add(tuple(sorted((i, j_mid))))
            #for j in intv_ids_p2: intv_pairs.add(tuple(sorted((i_mid, j))))
        for ij in intv_pairs:
            self.interval_pair_support[ij][signal_type] += count
            self.interval_pair_support_adj[ij].extend([[p1, p2]] * count)

    def process_adjacencies(self):
        for p1 in self.aln_index.adjacencies:
            for p2 in self.aln_index.adjacencies[p1]:
                for signal in self.aln_index.adjacencies[p1][p2]:
                    self.record_interval_pair_support(p1, p2, signal, self.aln_index.adjacencies[p1][p2][signal])

    def select_interval_pairs(self):
        evidence_counter = Counter()
        for interval_pair in sorted(self.interval_pair_support.keys()):
            support_by_signal_type = defaultdict(int)
            for i, group in enumerate([[SVSignal.CDEL, SVSignal.SRORD12],
                                       [SVSignal.SRORD21, SVSignal.SFORD21],
                                       [SVSignal.SRFLIP, SVSignal.SFFLIP]]):
                for signal in group:
                    support_by_signal_type[i] += self.interval_pair_support[interval_pair][signal]
            max_support = max(support_by_signal_type.values())
            if max_support < self.config.min_adj_support: continue
            # check if the adjacencies overlap
            adjacency_intervals = self.interval_pair_support_adj[interval_pair]
            adj_overlaps = NestedDict(list)
            for i in range(len(adjacency_intervals)):
                for j in range(i + 1, len(adjacency_intervals)):
                    if max(adjacency_intervals[i][0], adjacency_intervals[j][0]) < \
                            min(adjacency_intervals[i][1], adjacency_intervals[j][1]):
                        adj_overlaps[i].append(j)
                        adj_overlaps[j].append(i)
            if not any(len(v) + 1 >= self.config.min_adj_support for v in adj_overlaps.values()): continue
            self.interval_pairs.append(interval_pair)
            evidence_counter.update([max_support])
        logging.info("Selected %d interval pairs out of %d pairs for %s" % (len(self.interval_pairs),
                                                                     len(self.interval_pair_support), self.chr.name))

    def record_intervals(self):
        intervals = set()
        for x_id, y_id in tqdm(self.interval_pairs):
            intervals.add(x_id)
            intervals.add(y_id)
        intervals = sorted(list(intervals))
        itv_range = [intervals[0], intervals[0]]
        for i in intervals[1:]:
            if (itv_range[1] + self.steps_per_interval - 1) >= i:
                itv_range[1] = i
            else:
                self.intervals.append((itv_range[0] * self.step_size,
                                       itv_range[-1] * self.step_size + self.interval_size))
                itv_range = [i, i]
        self.intervals.append((itv_range[0] * self.step_size, itv_range[-1] * self.step_size + self.interval_size))

    def get_interval_ids(self, pos):
        start_interval_id = pos // self.config.step_size
        return [start_interval_id - i for i in range(self.steps_per_interval) if start_interval_id - i >= 0]

    def __iter__(self):
        for x_id, y_id in tqdm(self.interval_pairs):
            x = x_id * self.step_size
            y = y_id * self.step_size
            if x + self.interval_size > self.chr.len or y + self.interval_size > self.chr.len: continue 
            self.log_intervals(x, y)           
            yield GenomeIntervalPair(GenomeInterval(self.chr.tid, x, x + self.interval_size),
                                     GenomeInterval(self.chr.tid, y, y + self.interval_size))
