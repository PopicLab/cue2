from collections import defaultdict
import intervaltree

class GenomeInterval(tuple):
    def __new__(cls, chr_tid, start, end):
        return tuple.__new__(GenomeInterval, (chr_tid, start, end))

    def __init__(self, chr_tid, start, end):
        self.chr_tid = chr_tid
        self.start = start
        self.end = end

    def __len__(self):
        return self.end - self.start

    def __str__(self):
        return "%s_%d_%d" % (self.chr_tid, self.start, self.end)

    def __lt__(self, interval):
        return self.start < interval.start

    def to_list(self):
        return [self.chr_tid, self.start, self.end]

    @staticmethod
    def from_list(interval_list):
        return GenomeInterval(interval_list[0], interval_list[1], interval_list[2])


class GenomeIntervalPair:
    def __init__(self, intervalA, intervalB):
        self.intervalA = intervalA
        self.intervalB = intervalB

    def __str__(self):
        return "%s_&_%s" % (str(self.intervalA), str(self.intervalB))

    def to_list(self):
        return [self.intervalA.to_list(), self.intervalB.to_list()]

    @staticmethod
    def from_list(interval_pair_list):
        return GenomeIntervalPair(GenomeInterval.from_list(interval_pair_list[0]),
                                  GenomeInterval.from_list(interval_pair_list[1]))


class SVIntervalTree:
    def __init__(self, intervals):
        self.chr2tree = defaultdict(intervaltree.IntervalTree)
        for interval in intervals:
            if not self.overlaps(interval):
                self.add(interval)

    def overlaps(self, sv):
        if not self.chr2tree[sv.chr_name].overlaps(sv.start, sv.end): return []
        overlap = []
        for c in self.chr2tree[sv.chr_name].overlap(sv.start, sv.end):
            overlap_start = max(c.data.start, sv.start)
            overlap_end = min(c.data.end, sv.end)
            candidate_len = c.data.end - c.data.start
            if overlap_start < overlap_end:
                overlap_frac = float((overlap_end - overlap_start) / max(sv.len, candidate_len))
                size_frac = min(sv.len, candidate_len) / max(sv.len, candidate_len)
                overlap.append([c.data, overlap_frac, size_frac])
        return overlap

    def add(self, sv):
        self.chr2tree[sv.chr_name].addi(sv.start, sv.end, sv)

    def tree_by_chr(self, chr_name):
        return self.chr2tree[chr_name]
