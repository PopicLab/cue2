import cue.seq.intervals
import functools
import logging

def nms1D(svs, compare_func, overlap_frac=0.7, size_sim=0.7, low_evidence_thr=0.25):
    # filter out near-duplicate/overlapping calls (e.g. calls from different images that overlap the same interval)
    svs = sorted(svs, key=functools.cmp_to_key(compare_func), reverse=True)
    sv_tree = cue.seq.intervals.SVIntervalTree([])
    results = []
    for sv in svs:
        keep = True
        for sv_alt, rec_overlap, size_frac in sv_tree.overlaps(sv):
            if size_frac >= size_sim or rec_overlap >= overlap_frac: keep = False
            if sv.evidence and sv.evidence/sv_alt.evidence < low_evidence_thr: keep = False
            if sv.evidence_fuzzy and not sv.evidence: keep = False
            if not keep: break
        if keep:
            sv_tree.add(sv)
            results.append(sv)
        else: logging.debug("FILTER nms1D: %s" % sv)
    return results
