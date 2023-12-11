#ifndef _OPERATOR_H
#define _OPERATOR_H

#include "linear_operator.h"

template< LinearOperator Matrix >
struct Operator {
  
  
  Operator() = delete;
  ~Operator() = default;

  template< typename Lambda > 
  void matvec(){
    static_cast<const Matrix*>(this)->matvec();
  }
  
}


#endif 