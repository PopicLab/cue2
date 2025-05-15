import cue.seq.io as io
import cue.utils.types
import bisect
from collections import defaultdict

TYPES_SINGLE_BKP = ["DEL", "INV", "DUP", "INVDUP"]
TYPES_MULTI_BKP = ["TRA"]
TYPES = TYPES_SINGLE_BKP + TYPES_MULTI_BKP
SVType = cue.utils.types.make_enum("SVType",
                                   ["NEG"] + TYPES +
                                   [sv + "_HOM" for sv in TYPES] +
                                   [sv + "_HET" for sv in TYPES], __name__)
SVTypeSet = cue.utils.types.make_enum("SVTypeSet",
                                      ["SINGLE5G"], __name__)
SV_CLASS_SETS_GT = {SVTypeSet.SINGLE5G}
TYPE_SET_TO_TYPES = {SVTypeSet.SINGLE5G: [SVType.NEG, SVType.DEL_HOM, SVType.INV_HOM, SVType.DUP_HOM,
                                          SVType.DEL_HET, SVType.INV_HET, SVType.DUP_HET,
                                          SVType.INVDUP_HOM, SVType.INVDUP_HET]}

class SV:
    def __init__(self, sv_type, chr_name, start, end, qual, gt):
        self.type = sv_type
        self.chr_name = chr_name
        self.start = start
        self.end = end
        self.qual = qual
        self.gt = gt
        self.len = abs(end - start)
        self.evidence = None
        self.evidence_fuzzy = None
        self.annotation = None
        self.assign_internal_type()

    def update(self, start, end, evidence, evidence_fuzzy, annotation):
        self.start = start
        self.end = end
        self.len = abs(end - start)
        self.evidence = evidence
        self.evidence_fuzzy = evidence_fuzzy
        self.annotation = annotation

    @classmethod
    def from_vcf(cls, rec):
        sv_len = abs(rec.stop - rec.pos)
        gt = (None, None)
        if 'GT' in rec.samples[rec.samples[0].name]:
            gt = rec.samples[rec.samples[0].name]['GT']
        assert 'SVTYPE' in rec.info
        sv = SV(rec.info['SVTYPE'], rec.contig, int(rec.pos) - 1, int(rec.pos) - 1 + sv_len, rec.qual, gt)
        if 'ReadSupport' in rec.info and rec.info['ReadSupport'] is not None:
            sv.evidence = int(rec.info['ReadSupport'])
        if 'ReadSupportFuzzy' in rec.info and rec.info['ReadSupportFuzzy'] is not None:
            sv.evidence_fuzzy = int(rec.info['ReadSupportFuzzy'])
        return sv

    @classmethod
    def from_bed(cls, rec):
        fields = rec.strip().split()
        assert len(fields) >= 9, "Unexpected number of fields in BED file (at least 9 must be present): %s" % rec
        chrA, startA, endA, chrB, startB, endB, type, _, gt = fields[:9]
        assert chrA == chrB, "Only breakpoints on the same chromosome are currently supported: %s" % rec
        assert gt in ["0/1", "1/0", "1/1"], "Unexpected genotype value: %s" % rec
        gt = gt.strip().split("/")
        return SV(type, chrA, int(startA), int(startB), None, (gt[0], gt[1]))

    def assign_internal_type(self):
        gt_token = "" if self.gt == (None, None) else "_HOM" if self.gt == (1, 1) else "_HET"
        self.internal_type = SVType[self.type + gt_token]

    @staticmethod
    def parse_internal_type(sv_class_enum):
        name_tokens = sv_class_enum.name.split("_")
        if len(name_tokens) == 2: return name_tokens[0], (1, 1) if name_tokens[1] == "HOM" else (0, 1)
        return name_tokens[0], (None, None)

    def __str__(self):
        return '\t'.join("{}: {}".format(k, v) for k, v in self.__dict__.items())

    @staticmethod
    def compare(sv1, sv2):
        return sv1.start - sv2.start

    @staticmethod
    def compare_by_score(sv1, sv2):
        return float(sv1.qual) - float(sv2.qual)

    @staticmethod
    def compare_by_support(sv1, sv2):
        return sv1.evidence - sv2.evidence

    def __lt__(self, sv):
        return self.start < sv.start

    @staticmethod
    def get_sv_type_labels(sv_classes):
        return {sv_class: i for i, sv_class in enumerate(sv_classes)}


class SVContainer:
    def __init__(self, sv_gt_fname, chr_index):
        self.chr2rec = defaultdict(list)
        self.chr2starts = defaultdict(list)
        self.chr2ends = defaultdict(list)
        iterator = io.bed_iter(sv_gt_fname) if sv_gt_fname.endswith("bed") else io.vcf_iter(sv_gt_fname)
        for i, sv in enumerate(iterator):
            chr_tid = chr_index.get_chr_by_name(sv.chr_name).tid
            self.chr2rec[chr_tid].append(sv)
            self.chr2starts[chr_tid].append((sv.start, len(self.chr2rec[chr_tid]) - 1))
            self.chr2ends[chr_tid].append((sv.end, len(self.chr2rec[chr_tid]) - 1))
        for chr_tid in self.chr2rec:
            self.chr2starts[chr_tid] = sorted(self.chr2starts[chr_tid])
            self.chr2ends[chr_tid] = sorted(self.chr2ends[chr_tid])

    def size(self):
        return sum([len(self.chr2rec[r]) for r in self.chr2rec])

    def coords_in_interval(self, interval, coords):
        idx_left = bisect.bisect_left(coords, (interval.start, 0))
        if idx_left >= len(coords): return []
        idx_right = bisect.bisect_left(coords, (interval.end, 0))
        return coords[idx_left:idx_right]

    def overlap(self, interval):
        starts = self.coords_in_interval(interval, self.chr2starts[interval.chr_tid])
        ends = self.coords_in_interval(interval, self.chr2ends[interval.chr_tid])
        records = set()
        for _, i in starts: records.add(self.chr2rec[interval.chr_tid][i])
        for _, i in ends: records.add(self.chr2rec[interval.chr_tid][i])
        return list(records)

    def contained(self, interval_start, interval_end):
        chr_tid = interval_start.chr_tid
        starts = self.coords_in_interval(interval_start, self.chr2starts[chr_tid])
        ends = self.coords_in_interval(interval_end, self.chr2ends[chr_tid])
        records = []
        for i in set([i for _, i in starts]).intersection([i for _, i in ends]):
            records.append(self.chr2rec[chr_tid][i])
        return records

    def __iter__(self):
        for chr_tid in self.chr2rec:
            for rec in self.chr2rec[chr_tid]:
                yield rec

