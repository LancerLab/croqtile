
#include <gtest/gtest.h>

#include <iostream>
#include <algorithm>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "utils/testing.h"
#include "utils/catz/macro.h"
#include "utils/catz/index.h"
#include "utils/catz/coord.h"
#include "utils/catz/trait.h"

using namespace catz;

class CatzIRTest : public ::testing::Test {
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
    temp_filename = temp_template; 
  }

  virtual void TearDown() {
    close(tempfile_desc);          
    remove(temp_filename.c_str()); 
  }

  void createFileWithContent(const std::string& filename,
                             const std::string& content) {
    std::ofstream out(filename);
    ASSERT_TRUE(out.is_open());
    out << content;
    out.close();
  }
};

TEST_F(CatzIRTest, TraitTest) {
  const int compileTimeValue = 42;
  auto indexConst = make_index<compileTimeValue>();
  SCHECK(is_index_like_v<decltype(indexConst)> == true);

  int runTimeValue = 42;
  auto indexDyn = make_index(runTimeValue); 
  SCHECK(is_index_like_v<int> == false);
  SCHECK(is_index_like_v<decltype(indexDyn)> == true);
}

TEST_F(CatzIRTest, IndexCreateTest) {
  const int compileTimeValue = 42;
  auto indexConst = make_index<compileTimeValue>();
  SCHECK(indexConst.value == 42);
  SCHECK(indexConst.isStatic() == true);

  int runTimeValue = 42;
  auto indexDyn = make_index(runTimeValue); 
  CHECK(indexDyn.value == 42);
  SCHECK(indexDyn.isStatic() == false);
}

TEST_F(CatzIRTest, IndexSumTest) {
  auto idx1 = make_index<10>(); 
  auto idx2 = make_index<3>();  
  auto sum = idx1 + idx2;
  SCHECK(sum.value == 13);

  auto idx1_dyn = make_index(10);
  auto idx2_dyn = make_index(3); 
  auto sum_dyn = idx1 + idx2;
  CHECK(sum_dyn.value == 13);
}

TEST_F(CatzIRTest, IndexSubTest) {
  auto idx1 = make_index<10>(); 
  auto idx2 = make_index<3>();  
  auto res = idx1 - idx2;
  SCHECK(res.value == 7);

  auto idx1_dyn = make_index(10);
  auto idx2_dyn = make_index(3); 
  auto res_dyn = idx1 - idx2;
  CHECK(res_dyn.value == 7);
}

TEST_F(CatzIRTest, IndexMulTest) {
  auto idx1 = make_index<10>(); 
  auto idx2 = make_index<3>();  
  auto res = idx1 * idx2;
  SCHECK(res.value == 30);

  auto idx1_dyn = make_index(10); 
  auto idx2_dyn = make_index(3);   
  auto res_dyn = idx1 * idx2;
  CHECK(res_dyn.value == 30);
}

TEST_F(CatzIRTest, IndexDivTest) {
  auto idx1 = make_index<10>();  
  auto idx2 = make_index<3>();   
  auto res = idx1 / idx2;
  SCHECK(res.value == 3);

  auto idx1_dyn = make_index(10);
  auto idx2_dyn = make_index(3); 
  auto res_dyn = idx1 / idx2;
  CHECK(res_dyn.value == 3);
}

TEST_F(CatzIRTest, IndexModTest) {
  auto idx1 = make_index<10>();  
  auto idx2 = make_index<3>();   
  auto res = idx1 % idx2;
  SCHECK(res.value == 1);

  auto idx1_dyn = make_index(10);
  auto idx2_dyn = make_index(3); 
  auto res_dyn = idx1 % idx2;
  CHECK(res_dyn.value == 1);
}

TEST_F(CatzIRTest, IndexSelfIncrementTest) {
  auto idx1 = make_index<10>();  
  auto idx2 = ++idx1;
  SCHECK(idx1.value == 10);
  SCHECK(idx2.value == 11);

  auto idx1_dyn = make_index(10);  
  ++idx1_dyn;
  CHECK(idx1_dyn.value == 11);
}

TEST_F(CatzIRTest, CoordCreateTest) {
  auto r = make_index<2>();
  auto c = make_index<3>();
  auto static_coord = Coord(r, c);
  SCHECK(static_coord.rows() == 2);
  SCHECK(static_coord.cols() == 3);

  auto r_dyn = make_index(5);
  auto c_dyn = make_index(6);
  auto dynamic_coord = Coord(r_dyn, c_dyn); 
  CHECK(dynamic_coord.rows() == 5);
  CHECK(dynamic_coord.cols() == 6);
}

TEST_F(CatzIRTest, CoordCreate2Test) {
  auto static_coord = make_coord(2, 3);
  SCHECK(static_coord.rows() == 2);
  SCHECK(static_coord.cols() == 3);
  SCHECK(static_coord.isStatic() == true);

  int r_dyn = 5;
  int c_dyn = 6;
  auto dynamic_coord = make_coord_dyn(r_dyn, c_dyn); 
  CHECK(dynamic_coord.rows() == 5);
  CHECK(dynamic_coord.cols() == 6);
  SCHECK(dynamic_coord.isStatic() == false);

  const int r_2 = 5;
  const int c_2 = 6;
  auto static_coord_2 = make_coord(r_2, c_2); 
  SCHECK(static_coord_2.rows() == 5);
  SCHECK(static_coord_2.cols() == 6);
  SCHECK(static_coord_2.isStatic() == true);

  constexpr int r_3 = 5;
  constexpr int c_3 = 6;
  auto static_coord_3 = make_coord(r_3, c_3); 
  SCHECK(static_coord_3.rows() == 5);
  SCHECK(static_coord_3.cols() == 6);
  SCHECK(static_coord_3.isStatic() == true);
}

TEST_F(CatzIRTest, CoordSumTest) {
  const int r = 2;
  const int c = 3;
  auto coord1 = make_coord(r, c); 
  auto coord2 = make_coord(4, 5);
  auto coord3 = coord1 + coord2;

  SCHECK(coord1.isStatic() == true);

  SCHECK(coord3.rows() == 6);
  SCHECK(coord3.cols() == 8);
}

TEST_F(CatzIRTest, CoordSubTest) {
  constexpr int r = 2;
  constexpr int c = 3;
  auto coord1 = make_coord(r, c); 
  auto coord2 = make_coord(4, 5);
  auto coord3 = coord1 - coord2;

  SCHECK(coord3.isStatic() == true);

  SCHECK(coord3.rows() == -2);
  SCHECK(coord3.cols() == -2);
}

TEST_F(CatzIRTest, CoordMulTest) {
  constexpr int r = 2;
  constexpr int c = 3;
  auto coord1 = make_coord(r, c);
  auto coord2 = make_coord(4, 5);
  auto coord3 = coord1 * coord2;

  SCHECK(coord3.isStatic() == true);

  SCHECK(coord3.rows() == 8);
  SCHECK(coord3.cols() == 15);
}

TEST_F(CatzIRTest, CoordModTest) {
  constexpr int r = 2;
  constexpr int c = 3;
  auto coord1 = make_coord(r, c); 
  auto coord2 = make_coord(4, 5);
  auto coord3 = coord1 % coord2;

  SCHECK(coord3.isStatic() == true);

  SCHECK(coord3.rows() == 2);
  SCHECK(coord3.cols() == 3);
}

TEST_F(CatzIRTest, CoordCeilTest) {
  constexpr int r = 2;
  constexpr int c = 3;
  auto coord1 = make_coord(r, c);
  auto coord2 = make_coord(4, 5);
  auto coord3 = coord1.ceil_div(coord2);

  SCHECK(coord1.isStatic() == true);
  SCHECK(coord2.isStatic() == true);
  SCHECK(coord3.isStatic() == true);

  SCHECK(coord3.rows() == 1);
  SCHECK(coord3.cols() == 1);
}
