"""Check public release boundaries and relative Markdown links (no GPU required)."""
import pathlib
import re
import sys
from urllib.parse import unquote, urlsplit

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    files = [p for p in ROOT.rglob('*') if p.is_file()
             and '.git' not in p.relative_to(ROOT).parts
             and '__pycache__' not in p.relative_to(ROOT).parts
             and not any(part.startswith('build') for part in p.relative_to(ROOT).parts)]
    for path in files:
        rel = path.relative_to(ROOT)
        if any(part in ('学习反馈', '原始聊天', '编辑草稿') for part in rel.parts):
            raise RuntimeError(f'Private/editorial directory included: {rel}')
        if re.match(r'^(?:0[89]|1\d|2\d)-', path.name):
            raise RuntimeError(f'Unfinished chapter included: {rel}')
        if path.suffix not in ('.md', '.cu', '.cpp', '.h', '.py', '.json', '.css', '.txt'):
            continue
        text = path.read_text(encoding='utf-8')
        if re.search(r'ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}', text):
            raise RuntimeError(f'Possible credential in {rel}')
        if path.suffix != '.md':
            continue
        # Ignore examples inside fenced code; inspect prose links only.
        prose = re.sub(r'^```.*?^```[^\n]*$', '', text, flags=re.M | re.S)
        if '[[' in prose:
            raise RuntimeError(f'Unconverted wikilink in {rel}')
        if 'chatgpt.com/share/' in prose or re.search(r'C:[/\\]Users[/\\]', prose):
            raise RuntimeError(f'Private source link or local home path in {rel}')
        for match in re.finditer(r'\]\((?:<([^>]+)>|([^\s)]+))\)', prose):
            url = match[1] or match[2]
            parts = urlsplit(url)
            if parts.scheme or parts.netloc or not parts.path:
                continue
            target = (path.parent / unquote(parts.path)).resolve()
            if not target.is_relative_to(ROOT) or not target.is_file():
                raise RuntimeError(f'Broken or external local link: {rel} -> {url}')
    notes = [p for p in files if p.parent == ROOT and re.match(r'^0[1-7]-', p.name)]
    if len(notes) != 7:
        raise RuntimeError('Expected seven published chapters')
    answers = (ROOT / '附录-C-练习答案.md').read_text(encoding='utf-8')
    if re.findall(r'^## 第 (\d+) 章', answers, re.M) != [str(i) for i in range(1, 8)]:
        raise RuntimeError('Answers must cover chapters 1 through 7 only')
    print(f'PASS public release: 7 chapters; {len(files)} files; relative links and privacy checks')
    print('NOT RUN: compilation, CUDA execution, or hardware diagnostics')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ValueError) as error:
        print(f'FAIL: {error}', file=sys.stderr)
        sys.exit(1)
