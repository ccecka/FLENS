      DOUBLE PRECISION FUNCTION DZNRM2( N, X, INCX )

      INTEGER             INCX, N
      DOUBLE COMPLEX      X(*)

      DOUBLE PRECISION    TEMP

      CALL DZNRM2_SUB( N, X, INCX, TEMP )
      DZNRM2 = TEMP

      RETURN
      END
