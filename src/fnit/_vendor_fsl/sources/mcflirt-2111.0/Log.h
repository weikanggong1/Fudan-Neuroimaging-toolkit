 /*  Log.h

    Mark Woolrich, FMRIB Image Analysis Group

    Copyright (C) 1999-2000 University of Oxford  */

/*  CCOPYRIGHT  */

#include <iostream>
#include <fstream>
#include <string>
#include "armawrap/newmat.h"

namespace UTILS{

#if !defined(__Log_h)
#define __Log_h

  class Log
  {
  public:
    static Log& getInstance();
    ~Log() { delete logger; }

    int establishDir(const std::string& name);
    void setDir(const std::string& name);
    void setLogFile(const std::string& name) {logfilename = name;}
    const std::string& getDir() const { return dir; }

    void out(const std::string& p_fname, const NEWMAT::Matrix& p_mat, bool p_addMatrixString = true);
    void out(const std::string& p_fname, const NEWMAT::RowVector& p_mat);
    void out(const std::string& p_fname, const NEWMAT::ColumnVector& p_mat);
    void in(const std::string& p_fname, NEWMAT::Matrix& p_mat);
    void in(const std::string& p_fname, NEWMAT::Matrix& p_mat, const int numRows, const int numWaves);
    void in(const std::string& p_fname, NEWMAT::RowVector& p_mat);
    void in(const std::string& p_fname, NEWMAT::ColumnVector& p_mat);

    std::ofstream& str() { return logfileout; }

  private:
    Log() {}

    const Log& operator=(Log&);
    Log(Log&);

    static Log* logger;
    std::string dir;
    std::ofstream logfileout;
    std::string logfilename;
  };

  inline void Log::out(const std::string& p_fname, const NEWMAT::Matrix& p_mat, bool p_addMatrixString)
  {
    std::ofstream out;
    out.open((dir + "/" + p_fname).c_str(), std::ios::out);

    if(p_addMatrixString)
      out << "/Matrix" << std::endl;

    out << p_mat;
    out.close();
  }

  inline void Log::out(const std::string& p_fname, const NEWMAT::ColumnVector& p_mat)
  {
    std::ofstream out;
    out.open((dir + "/" + p_fname).c_str(), std::ios::out);
    out << p_mat;
    out.close();
  }

  inline void Log::out(const std::string& p_fname, const NEWMAT::RowVector& p_mat)
  {
    std::ofstream out;
    out.open((dir + "/" + p_fname).c_str(), std::ios::out);
    out << p_mat;
    out.close();
  }

  inline void Log::in(const std::string& p_fname, NEWMAT::Matrix& p_mat)
  {
    std::ifstream in;
    in.open((dir + "/" + p_fname).c_str(), std::ios::in);

    if(!in)
      {
        std::cerr << "Unable to open " << p_fname << std::endl;
        throw;
      }

    int numWaves = 0;
    int numPoints = 0;

    std::string str;

    while(true)
      {
        in >> str;
        if(str == "/Matrix")
          break;
        else if(str == "/NumWaves")
          {
            in >> numWaves;
          }
        else if(str == "/NumPoints" || str == "/NumContrasts")
          {
            in >> numPoints;
          }
      }

    if(numWaves != 0)
      {
        p_mat.ReSize(numPoints, numWaves);
      }
    else
      {
        numWaves = p_mat.Ncols();
        numPoints = p_mat.Nrows();
      }

    for(int i = 1; i <= numPoints; i++)
      {
        for(int j = 1; j <= numWaves; j++)
          {
            in >> p_mat(i,j);
          }
	  }

    in.close();
  }

  inline void Log::in(const std::string& p_fname, NEWMAT::Matrix& p_mat, const int numWaves, const int numPoints)
  {
    std::ifstream in;
    in.open((dir + "/" + p_fname).c_str(), std::ios::in);

    if(!in)
      {
        std::cerr << "Unable to open " << p_fname << std::endl;
        throw;
      }

    p_mat.ReSize(numWaves, numPoints);

    for(int i = 1; i <= numPoints; i++)
      {
        for(int j = 1; j <= numWaves; j++)
          {
            in >> p_mat(i,j);
          }
	  }

    in.close();
  }

  inline void Log::in(const std::string& p_fname, NEWMAT::ColumnVector& p_mat)
  {
    int size = p_mat.Nrows();

    std::ifstream in;
    in.open((dir + "/" + p_fname).c_str(), std::ios::in);

    if(!in)
      {
        std::string err("Unable to open ");
        err =  err +  p_fname;
        throw NEWMAT::Exception(err.c_str());
      }

    for(int i = 1; i <= size; i++)
      {
        in >> p_mat(i);
      }

    in.close();
  }

  inline void Log::in(const std::string& p_fname, NEWMAT::RowVector& p_mat)
  {
    int size = p_mat.Ncols();

    std::ifstream in;
    in.open((dir + "/" + p_fname).c_str(), std::ios::in);

    if(!in)
      {
        std::string err("Unable to open ");
        err =  err +  p_fname;
        throw NEWMAT::Exception(err.c_str());
      }

    for(int i = 1; i <= size; i++)
      {
        in >> p_mat(i);
      }

    in.close();
  }

  inline Log& Log::getInstance(){
    if(logger == NULL)
      logger = new Log();

    return *logger;
  }

#endif

}
