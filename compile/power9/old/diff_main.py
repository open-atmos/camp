import difflib

def main():
  file1_path="log_cpu.txt"
  file2_path="log.txt"
  output_file_path="diff.txt"

  with open(file1_path) as f1:
    f1_text = f1.read()
  with open(file2_path) as f2:
    f2_text = f2.read()
  # Find and print the diff:
  for line in difflib.unified_diff(f1_text, f2_text, fromfile='file1', tofile='file2', lineterm=''):
    print(line)

def old29s():
  file1_path="log_cpu.txt"
  file2_path="log.txt"
  output_file_path="diff.txt"
  with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2, open(output_file_path, 'w') as output_file:
    d = difflib.Differ()
    diff = d.compare(file1.readlines(), file2.readlines())
    output_file.writelines(diff)

def old():
  file1_path="log_cpu.txt"
  file2_path="log.txt"
  output_file_path="diff.txt"

  with open(file1_path, 'r') as file1:
    lines1 = file1.readlines()

  with open(file2_path, 'r') as file2:
    lines2 = file2.readlines()

  d = difflib.Differ()
  diff = d.compare(lines1, lines2)

  with open(output_file_path, 'w') as output_file:
    output_file.writelines(diff)
