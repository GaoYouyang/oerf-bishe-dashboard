from paper_library.tools.build_pdf_term_index import term_patterns


def count_matches(term: dict, text: str) -> int:
    return sum(len(pattern.findall(text)) for pattern in term_patterns(term))


def test_pdf_matching_excludes_broad_search_keywords() -> None:
    term = {
        "zh": "support-fit 闭式融合",
        "en": "Closed-form support-consistency mixture",
        "aliases": ["support-fit", "support consistency weight", "closed-form mixture"],
        "keywords": "support fit; least squares; projection consistency; dual branch",
    }

    assert count_matches(term, "We solve the inverse problem by least squares.") == 0
    assert count_matches(term, "The closed-form mixture selects two experts.") == 1


def test_pdf_matching_keeps_curated_acronym_aliases() -> None:
    term = {
        "zh": "广义交叉验证",
        "en": "Generalized cross-validation",
        "aliases": ["GCV"],
        "keywords": "ridge; regularization; prediction risk",
    }

    assert count_matches(term, "GCV selects the regularization parameter.") == 1
    assert count_matches(term, "The ridge parameter controls regularization.") == 0


def test_pdf_matching_does_not_split_slash_terms_into_generic_words() -> None:
    term = {
        "zh": "支持/查询视角拆分",
        "en": "Support/query view split",
        "aliases": ["support view", "query view", "camera split"],
        "keywords": "support; query; camera",
    }

    assert count_matches(term, "Support vector machines use a separating margin.") == 0
    assert count_matches(term, "The support view is withheld from the query view.") == 2


def test_short_uppercase_acronyms_are_case_sensitive() -> None:
    term = {
        "zh": "ART / SIRT / SART",
        "en": "Algebraic reconstruction techniques",
        "aliases": ["ART", "SIRT", "SART"],
        "keywords": "iterative tomography",
    }

    assert count_matches(term, "A state-of-the-art classifier.") == 0
    assert count_matches(term, "ART and SIRT are iterative methods.") == 2
