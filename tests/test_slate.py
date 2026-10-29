import pytest
import slate


@pytest.mark.parametrize('full,short,match', [('Nikola Jokić', 'N. Jokic', True), ('Gary Trent Jr.', 'G. Trent', True),
                                            ('Jalen Williams', 'Jaylin Williams', False), ('Desmond Bane', 'D. Bane', True)])
def test_names_normalize_accents_and_suffixes_without_collapsing_full_names(full, short, match):
    assert slate.name_match(full, short) is match
