"""Gene-symbol matching through HGNC renames."""

import pytest

from plb.genes import HGNC, vocab_names
from plb.model import GeneMap

HGNC_TSV = (
    "hgnc_id\tsymbol\tprev_symbol\tensembl_gene_id\n"
    "HGNC:1\tHARS1\tHARS\tENSG0001\n"
    "HGNC:2\tTARS1\t\"TARS|TARS_OLD\"\tENSG0002\n"
    "HGNC:3\tNEWNAME\tOLDNAME\tENSG0003\n"
    "HGNC:4\tACTB\t\tENSG0004\n"
)


@pytest.fixture
def hgnc(tmp_path):
    p = tmp_path / "hgnc.txt"
    p.write_text(HGNC_TSV)
    return HGNC(p)


def test_candidates(hgnc):
    assert hgnc.candidates("HARS", "ENSG0001") == ["HARS", "HARS1"]
    assert hgnc.candidates("TARS") == ["TARS", "TARS1", "TARS_OLD"]
    assert hgnc.candidates("NEWNAME") == ["NEWNAME", "OLDNAME"]


def test_old_dataset_symbol_matches_new_vocab_symbol(hgnc):
    vocab = {"HARS1": 1, "TARS1": 2, "ACTB": 3}
    names = vocab_names(["HARS", "TARS", "ACTB", "UNKNOWN"], ["ENSG0001", "ENSG0002", "ENSG0004", "ENSG9"], vocab, hgnc)
    assert list(names) == ["HARS1", "TARS1", "ACTB", ""]


def test_new_dataset_symbol_matches_old_vocab_symbol(hgnc):
    assert list(vocab_names(["NEWNAME"], ["ENSG0003"], {"OLDNAME": 1}, hgnc)) == ["OLDNAME"]


def test_vocab_entry_used_once(hgnc):
    names = vocab_names(["HARS1", "HARS"], None, {"HARS1": 1}, hgnc)
    assert list(names) == ["HARS1", ""]


def test_without_hgnc_only_exact_matches():
    assert list(vocab_names(["HARS", "ACTB"], None, {"HARS1": 1, "ACTB": 2}, None)) == ["", "ACTB"]


def test_genemap_uses_matched_names(hgnc):
    vocab = {"HARS1": 7}
    gmap = GeneMap(["HARS", "X"], vocab, vocab_names(["HARS", "X"], None, vocab, hgnc))
    assert gmap.token.tolist() == [7, -1]
    assert gmap.pert_cols("HARS+ctrl") == [0]
    assert gmap.renamed == {"HARS": "HARS1"}
