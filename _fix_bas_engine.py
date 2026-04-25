import sys
with open(r'src\arena\bas_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix Indentation Error
content = content.replace(
    '        if not self.quiet:\n            print(f"\\n[BAS] Ingesting document \'{source_name}\' with auto-scaling agents...")',
    '        if not self.quiet:\n            print(f"\\n[BAS] Ingesting document \'{source_name}\' with auto-scaling agents...")'
)
content = content.replace(
    '        if not self.quiet:\n            print(f"  Document segmented into {len(segments)} blocks")',
    '        if not self.quiet:\n            print(f"  Document segmented into {len(segments)} blocks")'
)
# Wait, let me just fix the exact indentation logic.
# I'll use regex to make it fool proof.
import re
content = re.sub(
    r'(\s+)if not self\.quiet:\n\s+print\(f"\\n\[BAS\] Ingesting document .*',
    r'\1if not self.quiet:\n\1    print(f"\\n[BAS] Ingesting document \'{source_name}\' with auto-scaling agents...")',
    content
)
content = re.sub(
    r'(\s+)if not self\.quiet:\n\s+print\(f"  Document segmented into .*',
    r'\1if not self.quiet:\n\1    print(f"  Document segmented into {len(segments)} blocks")',
    content
)
content = re.sub(
    r'(\s+)if not self\.quiet:\n\s+print\(f"  ✓ Processed \'.*',
    r'\1if not self.quiet:\n\1    print(f"  ✓ Processed \'{source_name}\': {len(segments)} segments → {len(created_agents)} active agents")',
    content
)
content = re.sub(
    r'(\s+)if not self\.quiet:\n\s+print\(f"  ✓ BM25 Hybrid Lexical Memory updated"\)',
    r'\1if not self.quiet:\n\1    print(f"  ✓ BM25 Hybrid Lexical Memory updated")',
    content
)
content = re.sub(
    r'(\s+)if not self\.quiet:\n\s+print\(f"  ✓ Created \{len.*',
    r'\1if not self.quiet:\n\1    print(f"  ✓ Created {len(new_agents)} specialized agents")',
    content
)
content = re.sub(
    r'(\s+)if not self\.quiet:\n\s+print\(f"  ✓ Total agents in system: .*',
    r'\1if not self.quiet:\n\1    print(f"  ✓ Total agents in system: {len(self.segment_agents)}")',
    content
)
content = re.sub(
    r'(\s+)if not self\.quiet:\n\s+print\(f"  ✓ StudSar markers: .*',
    r'\1if not self.quiet:\n\1    print(f"  ✓ StudSar markers: {self.studsar.studsar_network.get_total_markers()}")',
    content
)

with open(r'src\arena\bas_engine.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Done fixing indentation')
