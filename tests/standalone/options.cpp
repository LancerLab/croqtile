#include "io.hpp"
#include <fcntl.h>
#include <fstream>
#include <gtest/gtest.h>
#include <unistd.h>

using namespace Choreo;

class OptionTest : public ::testing::Test {
protected:
  std::string temp_filename;
  int tempfile_desc;

  virtual void SetUp() {
    char temp_template[] = "/tmp/optiontest-XXXXXX";
    tempfile_desc = mkstemp(temp_template);
    if (tempfile_desc == -1) {
      perror("Error creating temporary file");
      exit(EXIT_FAILURE);
    }
    temp_filename = temp_template; // Update filename
  }

  virtual void TearDown() {
    close(tempfile_desc);          // Close file descriptor
    remove(temp_filename.c_str()); // Delete file
  }

  void createFileWithContent(const std::string& filename,
                             const std::string& content) {
    std::ofstream out(filename);
    ASSERT_TRUE(out.is_open());
    out << content;
    out.close();
  }
};

TEST_F(OptionTest, CorrectlyWritesToOutputFile) {
  OptionRegistry& registry = OptionRegistry::GetInstance();
  Option<std::string> outputPath(OptionKind::User, "--output", "-o", "",
                                 "Place the output into <file>.", "-o <file>",
                                 true);

  const char* argv[] = {"program", "--output", temp_filename.c_str(), "-"};
  int argc = sizeof(argv) / sizeof(argv[0]);

  ASSERT_TRUE(registry.Parse(argc, const_cast<char**>(argv)));
  ASSERT_EQ(temp_filename, outputPath.GetValue());
  ASSERT_EQ(registry.StdinAsInput(), true);

  createFileWithContent(temp_filename, "Test output content");

  std::ifstream inFile(temp_filename);
  std::string fileContent;
  std::getline(inFile, fileContent);
  inFile.close();

  ASSERT_EQ("Test output content", fileContent);
}

TEST_F(OptionTest, HandlesMissingArguments) {
  OptionRegistry& registry = OptionRegistry::GetInstance();
  Option<std::string> criticalOption(OptionKind::User, "--critical", "-c", "",
                                     "a", "b", true);

  const char* argv[] = {"program", "--critical"};
  int argc = sizeof(argv) / sizeof(argv[0]);

  ASSERT_FALSE(registry.Parse(argc, const_cast<char**>(argv)));
}

TEST_F(OptionTest, DescriptionPrint) {
  OptionRegistry& registry = OptionRegistry::GetInstance();
  Option<std::string> whatOption(OptionKind::User, "--what", "-w", "",
                                 "I don't know what it is.", "", true);

  const char* argv[] = {"program", "--help"};
  int argc = sizeof(argv) / sizeof(argv[0]);

  ASSERT_EQ(registry.Parse(argc, const_cast<char**>(argv)), false);
  ASSERT_EQ(registry.Message(), std::string(""));
  ASSERT_EQ(registry.ReturnCode(), 0);
}
