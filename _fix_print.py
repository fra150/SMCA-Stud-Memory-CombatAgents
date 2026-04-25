import sys
with open(r'src\managers\manager.py', 'r', encoding='utf-8') as f:
    content = f.read()
old = '        print(f"StudSarManager will use device: {self.device}")'
new = '        if not self.quiet:\n            print(f"StudSarManager will use device: {self.device}")'
content = content.replace(old, new)
with open(r'src\managers\manager.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Done')
