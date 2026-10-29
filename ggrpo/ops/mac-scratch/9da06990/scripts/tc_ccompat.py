import re, sys
strict = re.compile(r'\b(std|vector|string|map|set|unordered_map|unordered_set|pair|queue|stack|deque|priority_queue|list|cout|cin|endl|new|delete|class|template|auto|nullptr|sort|reverse|swap|max_element|min_element|accumulate|find|count|lower_bound|upper_bound|next_permutation|make_pair|push_back|size|length|substr|bitset|to_string|stoi|INT_MAX|INT_MIN|iterator|begin|end|fill|binary_search|__gcd|abs|pow|sqrt|max|min|bool|true|false)\b|&\s*[a-z_]+\s*[,)]|::')
hard = re.compile(r'\b(std|vector|string|map|set|unordered_map|unordered_set|pair|queue|stack|deque|priority_queue|list|cout|cin|new|delete|class|template|auto|sort|reverse|max_element|min_element|accumulate|find|lower_bound|upper_bound|next_permutation|make_pair|push_back|substr|bitset|to_string|stoi|iterator|begin|end|binary_search|__gcd)\b|::|\.\s*(size|length|push_back|begin|end)\b|[A-Za-z_]\w*\s*&\s*\w+\s*[,)]')
tot=0; ok_hard=0
for path in sys.argv[1:]:
    for line in open(path, encoding='utf-8', errors='replace'):
        if '|' not in line: continue
        fid, code = line.split('|', 1)
        tot += 1
        if not hard.search(code): ok_hard += 1
print('functions', tot, 'no STL/C++-only constructs (heuristic)', ok_hard)
