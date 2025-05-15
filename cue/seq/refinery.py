from cue.seq.constants import SVSignal
from cue.seq.utils import *
from cue.utils.types import NestedDict
from cue.seq.index import CueIndex
from intervaltree import IntervalTree
import numpy as np

EVIDENCE_TYPES = {"DEL": [SVSignal.CDEL, SVSignal.SRORD12],
                  "DUP": [SVSignal.SRORD21, SVSignal.SFORD21],
                  "INV": [SVSignal.SRFLIP, SVSignal.SFFLIP],
                  "INVDUP": [SVSignal.SRFLIP, SVSignal.SFFLIP, SVSignal.SRORD21, SVSignal.SFORD21],
                  }

def refine_svs(chr_name, svs, config, log_fname):
    log_file = open(log_fname, 'w')
    aln_index = CueIndex.generate_or_load(chr_name, config)
    novel_adj = IntervalTree()
    for p1 in aln_index.adjacencies.keys():
        for p2 in aln_index.adjacencies[p1].keys():
            if p2 > p1: novel_adj.addi(p1, p2, aln_index.adjacencies[p1][p2])

    svs_refined = []
    for sv in svs:
        print("------IN: %s" % sv, file=log_file)
        is_valid = refine_sv(sv, novel_adj, config, log_file)
        print("------OUT: %s %s" % (is_valid, sv), file=log_file)
        if is_valid: svs_refined.append(sv)
    log_file.close()
    return svs_refined

def refine_sv(sv, adj_tree, config, log_file):
    adjacency_evidence = get_adjacencies(sv, adj_tree, config.refine_min_support)
    if not adjacency_evidence: return
    for evidence_type in EVIDENCE_TYPES[sv.type]:
        if evidence_type not in adjacency_evidence: continue
        for adj, count in adjacency_evidence[evidence_type].items():
            adjacency_evidence["all"][adj] += count
    print(adjacency_evidence, file=log_file)
    fuzzy_support = sum(adjacency_evidence["all"].values())
    for evidence_type in EVIDENCE_TYPES[sv.type] + ["all"]:
        if evidence_type not in adjacency_evidence: continue
        for adj, count in sorted(adjacency_evidence[evidence_type].items(), key=lambda item: item[1], reverse=True):
            if count < config.refine_min_support: continue
            p1, p2 = adj
            sv.update(int(p1), int(p2), count, fuzzy_support, "PRECISE")
            return True
    if fuzzy_support >= config.refine_min_support:
        p1 = np.mean([adj[0] for adj in adjacency_evidence["all"].keys()])
        p2 = np.mean([adj[1] for adj in adjacency_evidence["all"].keys()])
        sv.update(int(p1), int(p2), adjacency_evidence["all"][(p1, p2)], fuzzy_support, "IMPRECISE")
        return True
    return False

def get_adjacencies(sv, adj_tree, evidence_thr, size_sim=0.5, sv_min_size_check=500, bp_dist=1000):
    adj_by_size = []
    for c in adj_tree.overlap(sv.start, sv.end):
        if abs(c.begin - sv.start) > bp_dist or abs(c.end - sv.end) > bp_dist: continue
        adj_span = c.end - c.begin
        size_frac = min(adj_span, sv.len) / max(adj_span, sv.len)
        if (sv.len >= sv_min_size_check or adj_span >= sv_min_size_check) and size_frac < size_sim: continue
        adj_by_size.append([size_frac, c.data, c.begin, c.end])
    adj_dict = NestedDict(NestedDict(int))
    size_match = 0
    for adj in sorted(adj_by_size, key=lambda item: item[0], reverse=True):
        if adj[0] >= size_sim: size_match += 1
        if adj[0] < size_sim and size_match >= evidence_thr: continue
        for signal, count in adj[1].items():
            adj_dict[signal][(adj[2], adj[3])] = count
    return adj_dict

