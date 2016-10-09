#include <cmath>
#include <RIFeatures/struve.hpp>

namespace RIFeatures
{

/*      =============================================
!       Purpose: Compute Struve function H0(x)
!       Input :  x   --- Argument of H0(x)
!       Output:  SH0 --- H0(x)
!       ============================================= */
// Adpated from J-P Moreau
// http://jean-pierre.moreau.pagesperso-orange.fr/Cplus/mstvh0_cpp.txt
double struveh0(double X)
{
	double A0,BY0,P0,PI,Q0,R,S,T,T2,TA0,SH0;
	int K, KM;
	bool negative_X = (X < 0.0);
	if(negative_X)
		X = -X;

	PI=3.141592653589793;
	S=1.0;
	R=1.0;
	if (X <= 20.0)
	{
		A0=2.0*X/PI;
		for (K=1; K<61; K++)
		{
			R=-R*X/(2.0*K+1.0)*X/(2.0*K+1.0);
			S=S+R;
			if (std::abs(R) < std::abs(S)*1.0e-12) break;
		}
		SH0=A0*S;
	}
	else
	{
		KM=int(0.5*(X+1.0));
		if (X >= 50.0)
			KM=25;
		for (K=1; K<=KM; K++)
		{
			R=-R*std::pow((2.0*K-1.0)/X,2);
			S=S+R;
			if (std::abs(R) < std::abs(S)*1.0e-12)
				break;
		}
		T=4.0/X;
		T2=T*T;
		P0=((((-.37043e-5*T2+.173565e-4)*T2-.487613e-4)*T2+.17343e-3)*T2-0.1753062e-2)*T2+.3989422793;
		Q0=T*(((((.32312e-5*T2-0.142078e-4)*T2+0.342468e-4)*T2-0.869791e-4)*T2+0.4564324e-3)*T2-0.0124669441);
		TA0=X-0.25*PI;
		BY0=2.0/std::sqrt(X)*(P0*std::sin(TA0)+Q0*std::cos(TA0));
		SH0=2.0/(PI*X)*S+BY0;
	}

	// use odd symmetry around zero
	if(negative_X)
		SH0 = -SH0;
	return SH0;
}

double struveh1(double X)
{
	/*      =============================================
	!       Purpose: Compute Struve function H1(x)
	!       Input :  x   --- Argument of H1(x)
	!       Output:  SH1 --- H1(x)
	!       ============================================= */
	// Adpated from J-P Moreau
	// http://jean-pierre.moreau.pagesperso-orange.fr/Cplus/mstvh1_cpp.txt
	double A0,BY1,P1,PI,Q1,R,S,T,T2,TA1,SH1;
	int K, KM;

	// Use even symmetry around zero
	if(X < 0.0)
		X = -X;

	PI=3.141592653589793;
	R=1.0;
	if (X <= 20.0)
	{
		S=0.0;
		A0=-2.0/PI;
		for (K=1; K<=60; K++)
		{
			R=-R*X*X/(4.0*K*K-1.0);
			S=S+R;
			if (std::abs(R) < std::abs(S)*1.0e-12)
				break;
		}
		SH1=A0*S;
	}
	else
	{
		S=1.0;
		KM=int(0.5*X);
		if (X > 50.0)
			KM=25;
		for (K=1; K<=KM; K++)
		{
			R=-R*(4.0*K*K-1.0)/(X*X);
			S=S+R;
			if (std::abs(R) < std::abs(S)*1.0e-12)
				break;
		}
		T=4.0/X;
		T2=T*T;
		P1=((((0.42414e-5*T2-0.20092e-4)*T2+0.580759e-4)*T2-0.223203e-3)*T2+0.29218256e-2)*T2+0.3989422819;
		Q1=T*(((((-0.36594e-5*T2+0.1622e-4)*T2-0.398708e-4)*T2+0.1064741e-3)*T2-0.63904e-3)*T2+0.0374008364);
		TA1=X-0.75*PI;
		BY1=2.0/std::sqrt(X)*(P1*std::sin(TA1)+Q1*std::cos(TA1));
		SH1=2.0/PI*(1.0+S/(X*X))+BY1;
	}
	return SH1;
}

} // end of namespace
