from pathlib import Path
from gecko import reproducibility


def test_wheel_inside_unrelated_checkout_uses_current_source(monkeypatch, tmp_path):
    unrelated = tmp_path / 'old-checkout'
    selected = tmp_path / 'reviewer-snapshot'
    for root in (unrelated, selected):
        (root / 'src/gecko').mkdir(parents=True)
        (root / 'pyproject.toml').write_text('[project]\nname="gecko"\n')
    wheel_file = unrelated / '.venv/lib/python/site-packages/gecko/reproducibility.py'
    wheel_file.parent.mkdir(parents=True)
    wheel_file.write_text('')
    monkeypatch.setattr(reproducibility, '__file__', str(wheel_file))
    monkeypatch.chdir(selected)
    assert reproducibility.repository_root() == selected
