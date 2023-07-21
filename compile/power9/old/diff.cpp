#include <iostream>
#include <fstream>
#include <string>

bool compareFilesAndPrintDifferences(const std::string& file1_path, const std::string& file2_path, const std::string& output_file_path) {
  std::ifstream file1(file1_path);
  std::ifstream file2(file2_path);
  std::ofstream output_file(output_file_path);

  if (!file1.is_open() || !file2.is_open() || !output_file.is_open()) {
    std::cout << "Error opening files." << std::endl;
    return false;
  }

  std::string line1, line2;
  int line_number = 1;

  while (getline(file1, line1) && getline(file2, line2)) {
    if (line1 != line2) {
      output_file << "Line " << line_number << ":\n";
      output_file << "File1: " << line1 << "\n";
      output_file << "File2: " << line2 << "\n\n";
    }
    line_number++;
  }

  while (getline(file1, line1)) {
    output_file << "Line " << line_number << ":\n";
    output_file << "File1: " << line1 << "\n";
    output_file << "File2: <End of file>\n\n";
    line_number++;
  }

  while (getline(file2, line2)) {
    output_file << "Line " << line_number << ":\n";
    output_file << "File1: <End of file>\n";
    output_file << "File2: " << line2 << "\n\n";
    line_number++;
  }

  file1.close();
  file2.close();
  output_file.close();

  return true;
}

int main() {
  std::string file1_path = "log_cpu.txt";
  std::string file2_path = "log.txt";
  std::string output_file_path = "diff.txt";

  if (compareFilesAndPrintDifferences(file1_path, file2_path, output_file_path)) {
    std::cout << "Differences saved to output_diff.txt." << std::endl;
  } else {
    std::cout << "Error occurred during file comparison." << std::endl;
  }

  return 0;
}
