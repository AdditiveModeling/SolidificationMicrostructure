#include "gridutils.cpp"

void print(std::string stuff, int id) {
	std::cout << "Rank: " << id << ", " << stuff << "\n";
}

namespace MMSP
{

template <int dim, typename T> void update(grid<dim,vector<T> >& oldGrid, int steps)
{
	int id=0;
	int np=1;
	static int iterations=1;
	#ifdef MPI_VERSION
	id=MPI::COMM_WORLD.Get_rank();
	np=MPI::COMM_WORLD.Get_size();
	#endif
	

	ghostswap(oldGrid);
	
	vector<int> sampler(2,0);
	vector<int> sampler2(2,1);
	sampler2[1] = 2;
	vector<int> sampler3(2,2);
	sampler3[1] = 3;
	
	grid<dim,vector<T> > newGrid(oldGrid);
	grid<dim,vector<T> > dphi(oldGrid, dim); //face-centered delta phi grid
	grid<dim,vector<vector<T> > > dq(oldGrid, dim*2-2); //face-centered delta q_i, component (q1 thru q4), THEN direction (x,y,z)
	grid<dim,vector<T> > vdphi(oldGrid, dim); // face centered delta phi, but averaged about vertex (0.5, 0.5[, 0.5])
	grid<dim,vector<T> > vq(oldGrid, dim*2-2); //averaged values of q_i, about vertex (0.5, 0.5[, 0.5])
	grid<dim,vector<T> > vdpsi(oldGrid, dim); //values of delta psi, computed at vertex (0.5, 0.5[, 0.5])
	grid<dim,T> vmgphi2(oldGrid, 0); //vertex-centered mag-grad-phi-squared
	grid<dim,T> g(oldGrid, 0); //the interpolating function g = phi^2*(1-phi)^2
	grid<dim,T> gprime(oldGrid, 0); //the interpolating function g' = 4phi^3-6phi^2+2phi
	grid<dim,vector<T> > dpsi(oldGrid, dim); //values of delta psi, computed at cell (0, 0[, 0])
	grid<dim,vector<T> > pf_comp(oldGrid, 0); //components of the functional derivative w.r.t phi
	grid<dim,vector<T> > qf_comp(oldGrid, dim*2-2); //components of the functional derivative w.r.t q
	grid<dim,T> lphi(oldGrid, 0); //laplacian of phi, modified by the anisotropic factor (1-3y_e)
	grid<dim,T> mgq(oldGrid, 0); //mag-grad-q, *only* used within dF/dphi
	grid<dim,T> Q(oldGrid, 0); //a convenience grid for the rate equation of c. Unrelated to q.
	grid<dim,T> D_C(oldGrid, 0); //Chemical diffusion. Dependent on the phase, so its variable across the entire region
	
	//variables use cm,K,s, and J
	double  deltax=dx(oldGrid,0); //spacial step
	double      dt=5.29e-8;  // time-step
	double    temp=1574.;    //temperature
	double     y_e=0.12;     // degree of anisotropy
	double  M_qmax=80000000.;// maximum orientation mobility
	double       H=1e-11;    // Grain boundary energy coefficient
	double    T_mA=1728.;    // Melting point of Nickel (A)
	double    T_mB=1358.;    // Melting point of Copper (B)
	double     L_A=2350.;    // Latent heat of A
	double     L_B=1728.;    // Latent heat of B
	double     s_A=3.7e-5;   // Surface energy of A
	double     s_B=2.9e-5;   // Surface energy of B
	double     D_L=1e-5;     // Chemical diffusion in liquid phase
	double     D_S=1e-9;     // Chemical diffusion in solid phase
	double     B_A=0.33;     // Linear kinetic coefficient of A
	double     B_B=0.39;     // Linear kinetic coefficient of B
	double     v_m=7.42;     // Molar volume
	double       R=8.314;    // Gas constant
	double     d_i=deltax/0.94;  // Interfacial thickness
	double     s72=6*sqrt(2);// Convenience term to compute 6sqrt(2)
	double   ebar2=s72*s_A*d_i/T_mA;
	double  eqbar2=0.25*ebar2;
	double     W_A=3*s_A/(sqrt(2)*T_mA*d_i);
	double     W_B=3*s_B/(sqrt(2)*T_mB*d_i);
	double     M_A=T_mA*T_mA*B_A/(s72*L_A*d_i);
	double     M_B=T_mB*T_mB*B_B/(s72*L_B*d_i);

	std::cout.precision(2);

	int minus=0;
	int plus=0;
	for (int step=0; step<steps; ++step) {
		if (id==0)
			print_progress(step, steps);

		ghostswap(oldGrid);
		
		//2d sim, old (non-parallel) version. Will be removed soon
		/*
		if(dim == 2) {
			for (int i=0; i<nodes(oldGrid); ++i) {
				vector<int> x=position(oldGrid,i);
				
				// calculate grad(phi)
				vector<T> gradphi(dim,0.); // (0,0[,0])
				vector<vector<T> > gradq(2*dim-2, gradphi);
				for (int d=0; d<dim; ++d) {
					++x[d];
					T phir=oldGrid(x)[0];
					vector<T> qr(2*dim-2,0.);
					for(int j = 0; j < 2*dim-2; j++) {
						qr[j] = oldGrid(x)[j+2];
					}
					--x[d];
					gradphi[d]=(phir-oldGrid(x)[0])/deltax;
					for(int j = 0; j < 2*dim-2; j++) {
						gradq[j][d] = (qr[j]-oldGrid(x)[2+j])/deltax;
					}
				}
				dphi(i) = gradphi;
				dq(i) = gradq;
			
			}
			ghostswap(dphi);
			ghostswap(dq);
			
			for (int i=0; i<nodes(oldGrid); ++i) {
				vector<int> x=position(oldGrid,i);
				vector<T> q_av(2, 0.);
				vector<T> dphi_av(2, 0);
				dphi_av[0] += dphi(x)[0];
				dphi_av[1] += dphi(x)[1];
				for(int j = 0; j < 2; j++) {
					q_av[j] += oldGrid(x)[2+j];
				}
				++x[0];
				dphi_av[1] += dphi(x)[1];
				dphi_av[1] /= 2;
				for(int j = 0; j < 2; j++) {
					q_av[j] += oldGrid(x)[2+j];
				}
				++x[1];
				for(int j = 0; j < 2; j++) {
					q_av[j] += oldGrid(x)[2+j];
				}
				--x[0];
				dphi_av[0] += dphi(x)[0];
				dphi_av[0] /= 2;
				for(int j = 0; j < 2; j++) {
					q_av[j] += oldGrid(x)[2+j];
					q_av[j] /= 4;
				}
				vq(i) = q_av;
				vdphi(i) = dphi_av;
			}
			
			for (int i=0; i<nodes(oldGrid); ++i) {
				g(i) = oldGrid(i)[0]*oldGrid(i)[0]*(1-oldGrid(i)[0])*(1-oldGrid(i)[0]);
				gprime(i) = 4*oldGrid(i)[0]*oldGrid(i)[0]*oldGrid(i)[0]-6*oldGrid(i)[0]*oldGrid(i)[0]+2*oldGrid(i)[0];
				D_C(i) = D_L + (D_S-D_L)*oldGrid(i)[0]*oldGrid(i)[0]*oldGrid(i)[0]*(10-15*oldGrid(i)[0]+6*oldGrid(i)[0]*oldGrid(i)[0]);
				Q(i) = D_C(i)*v_m*oldGrid(i)[1]*(1-oldGrid(i)[1])*(gprime(i)*(W_B-W_A)+30*g(i)*(L_B*(1/T_mB-1/temp) - L_A*(1/T_mA-1/temp)))/R;
			
				T a2b2 = vq(i)[0]*vq(i)[0]-vq(i)[1]-vq(i)[1];
				T ab2 = 2*vq(i)[0]*vq(i)[1];
				vdpsi(i)[0] = a2b2*vdphi(i)[0] - ab2*vdphi(i)[1];
				vdpsi(i)[1] = a2b2*vdphi(i)[1] + ab2*vdphi(i)[0];
				vmgphi2(i) = vdphi(i)[0]*vdphi(i)[0]+vdphi(i)[1]*vdphi(i)[1]+0.000000000001; //add a trillionth to avoid divide-by-zero errors, for a negligible impact to accuracy
			}
			
			ghostswap(vq);
			ghostswap(vdphi);
			ghostswap(vdpsi);
			ghostswap(vmgphi2);
			ghostswap(D_C);
			ghostswap(Q);
			
			for (int i=0; i<nodes(oldGrid); ++i) {
				vector<int> x=position(oldGrid,i);
				vector<T> dpsi_av(2, 0.);
				vector<T> pfcomp_av(2, 0.);
				vector<T> dqs_av(2, 0.);
				vector<T> qfcomp_av(2, 0.);
				T _lphi = 0;
				T psix3 = vdpsi(x)[0]*vdpsi(x)[0]*vdpsi(x)[0];
				T psiy3 = vdpsi(x)[1]*vdpsi(x)[1]*vdpsi(x)[1];
				T mgphi4 = vmgphi2(x)*vmgphi2(x); 
				qfcomp_av[0] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[0]*vdphi(x)[0] - vq(x)[1]*vdphi(x)[1])+psiy3*(vq(x)[0]*vdphi(x)[1] + vq(x)[1]*vdphi(x)[0]));
				qfcomp_av[1] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(-vq(x)[1]*vdphi(x)[0] - vq(x)[0]*vdphi(x)[1])+psiy3*(vq(x)[0]*vdphi(x)[0] - vq(x)[1]*vdphi(x)[1]));
				pfcomp_av[0] += 4*y_e*((2*psix3*(vq(x)[0]*vq(x)[0]-vq(x)[1]*vq(x)[1]) + 4*psiy3*vq(x)[0]*vq(x)[1])/vmgphi2(x) - (vdphi(x)[0]*(psix3*vdpsi(x)[0] + psiy3*vdpsi(x)[1]))/mgphi4);
				pfcomp_av[1] += 4*y_e*((2*psiy3*(vq(x)[0]*vq(x)[0]-vq(x)[1]*vq(x)[1]) - 4*psix3*vq(x)[0]*vq(x)[1])/vmgphi2(x) - (vdphi(x)[1]*(psix3*vdpsi(x)[0] + psiy3*vdpsi(x)[1]))/mgphi4);
				_lphi += dphi(x)[0]*(1-3*y_e);
				_lphi += dphi(x)[1]*(1-3*y_e);
				for(int d = 0; d < 2; d++) {
					dpsi_av[d] += vdpsi(x)[d];
					for(int j = 0; j < 2; j++) {
						dqs_av[d] += dq(x)[j][d]*dq(x)[j][d];
					}
				}
				--x[0];
				for(int j = 0; j < 2; j++) {
					dqs_av[0] += dq(x)[j][0]*dq(x)[j][0];
				}
				psix3 = vdpsi(x)[0]*vdpsi(x)[0]*vdpsi(x)[0];
				psiy3 = vdpsi(x)[1]*vdpsi(x)[1]*vdpsi(x)[1];
				mgphi4 = vmgphi2(x)*vmgphi2(x);
				qfcomp_av[0] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[0]*vdphi(x)[0] - vq(x)[1]*vdphi(x)[1])+psiy3*(vq(x)[0]*vdphi(x)[1] + vq(x)[1]*vdphi(x)[0]));
				qfcomp_av[1] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(-vq(x)[1]*vdphi(x)[0] - vq(x)[0]*vdphi(x)[1])+psiy3*(vq(x)[0]*vdphi(x)[0] - vq(x)[1]*vdphi(x)[1]));
				pfcomp_av[0] -= 4*y_e*((2*psix3*(vq(x)[0]*vq(x)[0]-vq(x)[1]*vq(x)[1]) + 4*psiy3*vq(x)[0]*vq(x)[1])/vmgphi2(x) - (vdphi(x)[0]*(psix3*vdpsi(x)[0] + psiy3*vdpsi(x)[1]))/mgphi4);
				pfcomp_av[1] += 4*y_e*((2*psiy3*(vq(x)[0]*vq(x)[0]-vq(x)[1]*vq(x)[1]) - 4*psix3*vq(x)[0]*vq(x)[1])/vmgphi2(x) - (vdphi(x)[1]*(psix3*vdpsi(x)[0] + psiy3*vdpsi(x)[1]))/mgphi4);
				_lphi -= dphi(x)[0]*(1-3*y_e);
				for(int d = 0; d < 2; d++) {
					dpsi_av[d] += vdpsi(x)[d];
				}
				--x[1];
				psix3 = vdpsi(x)[0]*vdpsi(x)[0]*vdpsi(x)[0];
				psiy3 = vdpsi(x)[1]*vdpsi(x)[1]*vdpsi(x)[1];
				mgphi4 = vmgphi2(x)*vmgphi2(x);
				qfcomp_av[0] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[0]*vdphi(x)[0] - vq(x)[1]*vdphi(x)[1])+psiy3*(vq(x)[0]*vdphi(x)[1] + vq(x)[1]*vdphi(x)[0]));
				qfcomp_av[1] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(-vq(x)[1]*vdphi(x)[0] - vq(x)[0]*vdphi(x)[1])+psiy3*(vq(x)[0]*vdphi(x)[0] - vq(x)[1]*vdphi(x)[1]));
				pfcomp_av[0] -= 4*y_e*((2*psix3*(vq(x)[0]*vq(x)[0]-vq(x)[1]*vq(x)[1]) + 4*psiy3*vq(x)[0]*vq(x)[1])/vmgphi2(x) - (vdphi(x)[0]*(psix3*vdpsi(x)[0] + psiy3*vdpsi(x)[1]))/mgphi4);
				pfcomp_av[1] -= 4*y_e*((2*psiy3*(vq(x)[0]*vq(x)[0]-vq(x)[1]*vq(x)[1]) - 4*psix3*vq(x)[0]*vq(x)[1])/vmgphi2(x) - (vdphi(x)[1]*(psix3*vdpsi(x)[0] + psiy3*vdpsi(x)[1]))/mgphi4);
				for(int d = 0; d < 2; d++) {
					dpsi_av[d] += vdpsi(x)[d];
				}
				++x[0];
				for(int j = 0; j < 2; j++) {
					dqs_av[1] += dq(x)[j][1]*dq(x)[j][1];
				}
				psix3 = vdpsi(x)[0]*vdpsi(x)[0]*vdpsi(x)[0];
				psiy3 = vdpsi(x)[1]*vdpsi(x)[1]*vdpsi(x)[1];
				mgphi4 = vmgphi2(x)*vmgphi2(x);
				qfcomp_av[0] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[0]*vdphi(x)[0] - vq(x)[1]*vdphi(x)[1])+psiy3*(vq(x)[0]*vdphi(x)[1] + vq(x)[1]*vdphi(x)[0]));
				qfcomp_av[1] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(-vq(x)[1]*vdphi(x)[0] - vq(x)[0]*vdphi(x)[1])+psiy3*(vq(x)[0]*vdphi(x)[0] - vq(x)[1]*vdphi(x)[1]));
				pfcomp_av[0] += 4*y_e*((2*psix3*(vq(x)[0]*vq(x)[0]-vq(x)[1]*vq(x)[1]) + 4*psiy3*vq(x)[0]*vq(x)[1])/vmgphi2(x) - (vdphi(x)[0]*(psix3*vdpsi(x)[0] + psiy3*vdpsi(x)[1]))/mgphi4);
				pfcomp_av[1] -= 4*y_e*((2*psiy3*(vq(x)[0]*vq(x)[0]-vq(x)[1]*vq(x)[1]) - 4*psix3*vq(x)[0]*vq(x)[1])/vmgphi2(x) - (vdphi(x)[1]*(psix3*vdpsi(x)[0] + psiy3*vdpsi(x)[1]))/mgphi4);
				_lphi -= dphi(x)[1]*(1-3*y_e);
				for(int d = 0; d < 2; d++) {
					dpsi_av[d] += vdpsi(x)[d];
					dpsi_av[d] /= 4;
					pfcomp_av[d] /= (2.*deltax);
					qfcomp_av[d] /= 4;
				}
				lphi(i) = _lphi/deltax;
				mgq(i) = sqrt(0.5*(dqs_av[0] + dqs_av[1]));
				dpsi(i) = dpsi_av;
				pf_comp(i) = pfcomp_av;
				qf_comp(i) = qfcomp_av;
			}
			
			for (int i=0; i<nodes(oldGrid); ++i) {
				newGrid(i)[0] = oldGrid(i)[0] - dt*((1-oldGrid(i)[1])*M_A+oldGrid(i)[1]*M_B)*((1-oldGrid(i)[1])*(W_A*gprime(i)+30*(1/T_mA - 1/temp)*g(i)*L_A)+oldGrid(i)[1]*(W_B*gprime(i)+30*(1/T_mB - 1/temp)*g(i)*L_B) + 4*H*temp*oldGrid(i)[0]*mgq(i) - ebar2*(lphi(i)+pf_comp(i)[0]+pf_comp(i)[1]));
			}
			
			for (int i=0; i<nodes(oldGrid); ++i) {
				vector<int> x=position(oldGrid,i);
				T deltac = 0.;
				for(int d = 0; d < dim; d++) {
					--x[d];
					T phi_l = oldGrid(x)[0];
					T c_l = oldGrid(x)[1];
					T dc_l = D_C(x);
					T Q_l = Q(x);
					x[d] += 2;
					T phi_r = oldGrid(x)[0];
					T c_r = oldGrid(x)[1];
					T dc_r = D_C(x);
					T Q_r = Q(x);
					--x[d];
					deltac += 0.5*((dc_r + D_C(x))*(c_r-oldGrid(x)[1]) - (dc_l + D_C(x))*(oldGrid(x)[1]-c_l))/(deltax*deltax) + 0.5*((Q_r + Q(x))*(phi_r-oldGrid(x)[0]) - (Q_l + Q(x))*(oldGrid(x)[0]-phi_l))/(deltax*deltax);
				}
				newGrid(i)[1] = oldGrid(i)[1] + dt*deltac;
			}
			
			for (int i=0; i<nodes(oldGrid); ++i) {
				vector<int> x=position(oldGrid,i);
				vector<T> dq1_l(2, 0.); //"left" gradient of q1, index is x,y
				vector<T> dq4_l(2, 0.);
				vector<T> dq1_cc(2, 0.); //cell-centered gradient of q1
				vector<T> dq4_cc(2, 0.); //cell-centered gradient of q4
				vector<T> phi_l(2, 0.); //left value of phi
				vector<T> phi_r(2, 0.); //right value of phi
				vector<T> dphi_cc(2, 0.); //cell-centered gradient of phi
				vector<T> lq(2, 0.); //laplacian of q, index represents the quaternion index
				vector<T> rgqs_l(2, 0.); //the "left" root-grad-q-squared. Index says which direction (x,y) uses face-centered gradient to compute it
				vector<T> rgqs_r(2, 0.);
				vector<T> g_temp(2, 0.); //the value of the ugly expression div(p * grad(q_i)/rgqs), index is quaternion index
				T lmbda = 0.; //the lambda coefficient, comes from the lagrange multiplier.
				vector<T> deltaqi(2, 0.); //the rate equation w.r.t. time, for the various q_i components
				T M_q = 0.000001 + (M_qmax-0.000001)*(1-oldGrid(i)[0]*oldGrid(i)[0]*oldGrid(i)[0]*(10-15*oldGrid(i)[0]+6*oldGrid(i)[0]*oldGrid(i)[0]));
				
				for(int d = 0; d < 2; d++) {
					--x[d];
					phi_l[d] = oldGrid(x)[0];
					dq1_l[d] = dq(x)[0][d];
					dq4_l[d] = dq(x)[1][d];
					x[d] += 2;
					phi_r[d] = oldGrid(x)[0];
					dphi_cc[d] = 0.5*(phi_r[d]-phi_l[d])/deltax;
					--x[d];
					dq1_cc[d] = 0.5*(dq(x)[0][d]+dq1_l[d]);
					dq4_cc[d] = 0.5*(dq(x)[1][d]+dq4_l[d]);
					
				}
				
				lq[0] = ((dq(x)[0][0] - dq1_l[0]) + (dq(x)[0][1] - dq1_l[1]))/deltax;
				lq[1] = ((dq(x)[1][0] - dq4_l[0]) + (dq(x)[1][1] - dq4_l[1]))/deltax;
				rgqs_l[0] = max(sqrt(dq1_l[0]*dq1_l[0] + dq4_l[0]*dq4_l[0] + dq1_cc[1]*dq1_cc[1] + dq4_cc[1]*dq4_cc[1]), 1.);
				rgqs_l[1] = max(sqrt(dq1_l[1]*dq1_l[1] + dq4_l[1]*dq4_l[1] + dq1_cc[0]*dq1_cc[0] + dq4_cc[0]*dq4_cc[0]), 1.);
				rgqs_r[0] = max(sqrt(dq(x)[0][0]*dq(x)[0][0] + dq(x)[1][0]*dq(x)[1][0] + dq1_cc[1]*dq1_cc[1] + dq4_cc[1]*dq4_cc[1]), 1.);
				rgqs_r[1] = max(sqrt(dq(x)[0][1]*dq(x)[0][1] + dq(x)[1][1]*dq(x)[1][1] + dq1_cc[0]*dq1_cc[0] + dq4_cc[0]*dq4_cc[0]), 1.);
				for(int d = 0; d < 2; d++) {
					g_temp[0] += (dq(x)[0][d]*H*temp*(oldGrid(x)[0]*oldGrid(x)[0]+phi_r[d]*phi_r[d])/rgqs_r[d] - dq1_l[d]*H*temp*(oldGrid(x)[0]*oldGrid(x)[0]+phi_l[d]*phi_l[d])/rgqs_l[d]);
					g_temp[1] += (dq(x)[1][d]*H*temp*(oldGrid(x)[0]*oldGrid(x)[0]+phi_r[d]*phi_r[d])/rgqs_r[d] - dq4_l[d]*H*temp*(oldGrid(x)[0]*oldGrid(x)[0]+phi_l[d]*phi_l[d])/rgqs_l[d]);
				}
				T t1 = eqbar2*lq[0] + g_temp[0] + qf_comp(i)[0];
				T t4 = eqbar2*lq[1] + g_temp[1] + qf_comp(i)[1];
				lmbda = oldGrid(x)[2]*t1 + oldGrid(x)[3]*t4;
				deltaqi[0] = M_q*(t1 - oldGrid(x)[2]*lmbda);
				deltaqi[1] = M_q*(t4 - oldGrid(x)[3]*lmbda);
				
				newGrid(i)[2] = oldGrid(i)[2] + dt*deltaqi[0];
				newGrid(i)[3] = oldGrid(i)[3] + dt*deltaqi[1];
				
				//std::cout << qf_comp(i)[0] << ", " << qf_comp(i)[1] << "\n";
			}
			
			swap(oldGrid,newGrid);
		}
		*/
		
		//2d sim, new (parallelizable!) code
		if(dim == 2) {
			grid<dim, T> dpdx = partial_r(oldGrid, 0, 0, deltax);
			grid<dim, T> dpdy = partial_r(oldGrid, 0, 1, deltax);
			grid<dim, T> dq1dx = partial_r(oldGrid, 2, 0, deltax);
			grid<dim, T> dq1dy = partial_r(oldGrid, 2, 1, deltax);
			grid<dim, T> dq4dx = partial_r(oldGrid, 3, 0, deltax);
			grid<dim, T> dq4dy = partial_r(oldGrid, 3, 1, deltax);
			ghostswap(dpdx);
			ghostswap(dpdy);
			ghostswap(dq1dx);
			ghostswap(dq1dy);
			ghostswap(dq4dx);
			ghostswap(dq4dy);
			
			grid<dim, T> vdpdx = average_r(dpdx, 1);
			grid<dim, T> vdpdy = average_r(dpdy, 0);
			grid<dim, T> vq1x = average_r(oldGrid, 2, 0);
			ghostswap(vq1x);
			grid<dim, T> vq1 = average_r(vq1x, 1);
			ghostswap(vq1);
			grid<dim, T> vq4x = average_r(oldGrid, 3, 0);
			ghostswap(vq4x);
			grid<dim, T> vq4 = average_r(vq4x, 1);
			ghostswap(vq4);
			
			for (int i=0; i<nodes(oldGrid); ++i) {
				g(i) = oldGrid(i)[0]*oldGrid(i)[0]*(1-oldGrid(i)[0])*(1-oldGrid(i)[0]);
				gprime(i) = 4*oldGrid(i)[0]*oldGrid(i)[0]*oldGrid(i)[0]-6*oldGrid(i)[0]*oldGrid(i)[0]+2*oldGrid(i)[0];
				D_C(i) = D_L + (D_S-D_L)*oldGrid(i)[0]*oldGrid(i)[0]*oldGrid(i)[0]*(10-15*oldGrid(i)[0]+6*oldGrid(i)[0]*oldGrid(i)[0]);
				Q(i) = D_C(i)*v_m*oldGrid(i)[1]*(1-oldGrid(i)[1])*(gprime(i)*(W_B-W_A)+30*g(i)*(L_B*(1/T_mB-1/temp) - L_A*(1/T_mA-1/temp)))/R;
			
				T a2b2 = vq1(i)*vq1(i)-vq4(i)-vq4(i);
				T ab2 = 2*vq1(i)*vq4(i);
				vdpsi(i)[0] = a2b2*vdpdx(i) - ab2*vdpdy(i);
				vdpsi(i)[1] = a2b2*vdpdy(i) + ab2*vdpdx(i);
				vmgphi2(i) = vdpdx(i)*vdpdx(i)+vdpdy(i)*vdpdy(i)+0.000000000001; //add a trillionth to avoid divide-by-zero errors, for a negligible impact to accuracy
			}
			ghostswap(vdpsi);
			ghostswap(vmgphi2);
			ghostswap(D_C);
			ghostswap(Q);
			
			for (int i=0; i<nodes(oldGrid); ++i) {
				vector<int> x=position(oldGrid,i);
				vector<T> dpsi_av(2, 0.);
				vector<T> pfcomp_av(2, 0.);
				vector<T> dqs_av(2, 0.);
				vector<T> qfcomp_av(2, 0.);
				T _lphi = 0;
				T psix3 = vdpsi(x)[0]*vdpsi(x)[0]*vdpsi(x)[0];
				T psiy3 = vdpsi(x)[1]*vdpsi(x)[1]*vdpsi(x)[1];
				T mgphi4 = vmgphi2(x)*vmgphi2(x); 
				qfcomp_av[0] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq1(x)*vdpdx(x) - vq4(x)*vdpdy(x))+psiy3*(vq1(x)*vdpdy(x) + vq4(x)*vdpdx(x)));
				qfcomp_av[1] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(-vq4(x)*vdpdx(x) - vq1(x)*vdpdy(x))+psiy3*(vq1(x)*vdpdx(x) - vq4(x)*vdpdy(x)));
				pfcomp_av[0] += 4*y_e*((2*psix3*(vq1(x)*vq1(x)-vq4(x)*vq4(x)) + 4*psiy3*vq1(x)*vq4(x))/vmgphi2(x) - (vdpdx(x)*(psix3*vdpsi(x)[0] + psiy3*vdpsi(x)[1]))/mgphi4);
				pfcomp_av[1] += 4*y_e*((2*psiy3*(vq1(x)*vq1(x)-vq4(x)*vq4(x)) - 4*psix3*vq1(x)*vq4(x))/vmgphi2(x) - (vdpdy(x)*(psix3*vdpsi(x)[0] + psiy3*vdpsi(x)[1]))/mgphi4);
				_lphi += dpdx(x)*(1-3*y_e);
				_lphi += dpdy(x)*(1-3*y_e);
				dqs_av[0] += dq1dx(x)*dq1dx(x);
				dqs_av[0] += dq4dx(x)*dq4dx(x);
				dqs_av[1] += dq1dy(x)*dq1dy(x);
				dqs_av[1] += dq4dy(x)*dq4dy(x);
				--x[0];
				dqs_av[0] += dq1dx(x)*dq1dx(x);
				dqs_av[0] += dq4dx(x)*dq4dx(x);
				_lphi -= dpdx(x)*(1-3*y_e);
				--x[1];
				++x[0];
				dqs_av[1] += dq1dy(x)*dq1dy(x);
				dqs_av[1] += dq4dy(x)*dq4dy(x);
				_lphi -= dpdy(x)*(1-3*y_e);
				lphi(i) = _lphi/deltax;
				mgq(i) = sqrt(0.5*(dqs_av[0] + dqs_av[1]));
				pf_comp(i) = pfcomp_av;
				qf_comp(i) = qfcomp_av;
			}
			ghostswap(pf_comp);
			grid<dim, T> pfc_x_temp = partial_l(pf_comp, 0, 0, deltax);
			ghostswap(pfc_x_temp);
			grid<dim, T> pfc_x = average_l(pfc_x_temp, 1);
			grid<dim, T> pfc_y_temp = partial_l(pf_comp, 1, 1, deltax);
			ghostswap(pfc_y_temp);
			grid<dim, T> pfc_y = average_l(pfc_y_temp, 0);
			ghostswap(qf_comp);
			grid<dim, T> qfc_x_temp = average_l(qf_comp, 0, 0);
			ghostswap(qfc_x_temp);
			grid<dim, T> qfc_x = average_l(qfc_x_temp, 1);
			grid<dim, T> qfc_y_temp = average_l(qf_comp, 1, 1);
			ghostswap(qfc_y_temp);
			grid<dim, T> qfc_y = average_l(qfc_y_temp, 0);
			
			for (int i=0; i<nodes(oldGrid); ++i) {
				newGrid(i)[0] = oldGrid(i)[0] - dt*((1-oldGrid(i)[1])*M_A+oldGrid(i)[1]*M_B)*((1-oldGrid(i)[1])*(W_A*gprime(i)+30*(1/T_mA - 1/temp)*g(i)*L_A)+oldGrid(i)[1]*(W_B*gprime(i)+30*(1/T_mB - 1/temp)*g(i)*L_B) + 4*H*temp*oldGrid(i)[0]*mgq(i) - ebar2*(lphi(i)+pfc_x(i)+pfc_y(i)));
			}
			
			for (int i=0; i<nodes(oldGrid); ++i) {
				vector<int> x=position(oldGrid,i);
				T deltac = 0.;
				for(int d = 0; d < dim; d++) {
					--x[d];
					T phi_l = oldGrid(x)[0];
					T c_l = oldGrid(x)[1];
					T dc_l = D_C(x);
					T Q_l = Q(x);
					x[d] += 2;
					T phi_r = oldGrid(x)[0];
					T c_r = oldGrid(x)[1];
					T dc_r = D_C(x);
					T Q_r = Q(x);
					--x[d];
					deltac += 0.5*((dc_r + D_C(x))*(c_r-oldGrid(x)[1]) - (dc_l + D_C(x))*(oldGrid(x)[1]-c_l))/(deltax*deltax) + 0.5*((Q_r + Q(x))*(phi_r-oldGrid(x)[0]) - (Q_l + Q(x))*(oldGrid(x)[0]-phi_l))/(deltax*deltax);
				}
				newGrid(i)[1] = oldGrid(i)[1] + dt*deltac;
			}
			
			for (int i=0; i<nodes(oldGrid); ++i) {
				vector<int> x=position(oldGrid,i);
				vector<T> dq1_l(2, 0.); //"left" gradient of q1, index is x,y
				vector<T> dq4_l(2, 0.);
				vector<T> dq1_cc(2, 0.); //cell-centered gradient of q1
				vector<T> dq4_cc(2, 0.); //cell-centered gradient of q4
				vector<T> phi_l(2, 0.); //left value of phi
				vector<T> phi_r(2, 0.); //right value of phi
				vector<T> dphi_cc(2, 0.); //cell-centered gradient of phi
				vector<T> lq(2, 0.); //laplacian of q, index represents the quaternion index
				vector<T> rgqs_l(2, 0.); //the "left" root-grad-q-squared. Index says which direction (x,y) uses face-centered gradient to compute it
				vector<T> rgqs_r(2, 0.);
				vector<T> g_temp(2, 0.); //the value of the ugly expression div(p * grad(q_i)/rgqs), index is quaternion index
				T lmbda = 0.; //the lambda coefficient, comes from the lagrange multiplier.
				vector<T> deltaqi(2, 0.); //the rate equation w.r.t. time, for the various q_i components
				T M_q = 0.000001 + (M_qmax-0.000001)*(1-oldGrid(i)[0]*oldGrid(i)[0]*oldGrid(i)[0]*(10-15*oldGrid(i)[0]+6*oldGrid(i)[0]*oldGrid(i)[0]));
				
				--x[0];
				phi_l[0] = oldGrid(x)[0];
				dq1_l[0] = dq1dx(x);
				dq4_l[0] = dq4dx(x);
				x[0] += 2;
				phi_r[0] = oldGrid(x)[0];
				dphi_cc[0] = 0.5*(phi_r[0]-phi_l[0])/deltax;
				--x[0];
				dq1_cc[0] = 0.5*(dq1dx(x)+dq1_l[0]);
				dq4_cc[0] = 0.5*(dq4dx(x)+dq4_l[0]);
				
				--x[1];
				phi_l[1] = oldGrid(x)[0];
				dq1_l[1] = dq1dy(x);
				dq4_l[1] = dq4dy(x);
				x[1] += 2;
				phi_r[1] = oldGrid(x)[0];
				dphi_cc[1] = 0.5*(phi_r[1]-phi_l[1])/deltax;
				--x[1];
				dq1_cc[1] = 0.5*(dq1dy(x)+dq1_l[1]);
				dq4_cc[1] = 0.5*(dq4dy(x)+dq4_l[1]);
				
				lq[0] = ((dq1dx(x) - dq1_l[0]) + (dq1dy(x) - dq1_l[1]))/deltax;
				lq[1] = ((dq4dx(x) - dq4_l[0]) + (dq4dy(x) - dq4_l[1]))/deltax;
				rgqs_l[0] = max(sqrt(dq1_l[0]*dq1_l[0] + dq4_l[0]*dq4_l[0] + dq1_cc[1]*dq1_cc[1] + dq4_cc[1]*dq4_cc[1]), 1.);
				rgqs_l[1] = max(sqrt(dq1_l[1]*dq1_l[1] + dq4_l[1]*dq4_l[1] + dq1_cc[0]*dq1_cc[0] + dq4_cc[0]*dq4_cc[0]), 1.);
				rgqs_r[0] = max(sqrt(dq1dx(x)*dq1dx(x) + dq4dx(x)*dq4dx(x) + dq1_cc[1]*dq1_cc[1] + dq4_cc[1]*dq4_cc[1]), 1.);
				rgqs_r[1] = max(sqrt(dq1dy(x)*dq1dy(x) + dq4dy(x)*dq4dy(x) + dq1_cc[0]*dq1_cc[0] + dq4_cc[0]*dq4_cc[0]), 1.);
				g_temp[0] += (dq1dx(x)*H*temp*(oldGrid(x)[0]*oldGrid(x)[0]+phi_r[0]*phi_r[0])/rgqs_r[0] - dq1_l[0]*H*temp*(oldGrid(x)[0]*oldGrid(x)[0]+phi_l[0]*phi_l[0])/rgqs_l[0]);
				g_temp[1] += (dq4dx(x)*H*temp*(oldGrid(x)[0]*oldGrid(x)[0]+phi_r[0]*phi_r[0])/rgqs_r[0] - dq4_l[0]*H*temp*(oldGrid(x)[0]*oldGrid(x)[0]+phi_l[0]*phi_l[0])/rgqs_l[0]);
				g_temp[0] += (dq1dy(x)*H*temp*(oldGrid(x)[0]*oldGrid(x)[0]+phi_r[1]*phi_r[1])/rgqs_r[1] - dq1_l[1]*H*temp*(oldGrid(x)[0]*oldGrid(x)[0]+phi_l[1]*phi_l[1])/rgqs_l[1]);
				g_temp[1] += (dq4dy(x)*H*temp*(oldGrid(x)[0]*oldGrid(x)[0]+phi_r[1]*phi_r[1])/rgqs_r[1] - dq4_l[1]*H*temp*(oldGrid(x)[0]*oldGrid(x)[0]+phi_l[1]*phi_l[1])/rgqs_l[1]);
				T t1 = eqbar2*lq[0] + g_temp[0] + qfc_x(i);
				T t4 = eqbar2*lq[1] + g_temp[1] + qfc_y(i);
				lmbda = oldGrid(x)[2]*t1 + oldGrid(x)[3]*t4;
				deltaqi[0] = M_q*(t1 - oldGrid(x)[2]*lmbda);
				deltaqi[1] = M_q*(t4 - oldGrid(x)[3]*lmbda);
				
				newGrid(i)[2] = oldGrid(i)[2] + dt*deltaqi[0];
				newGrid(i)[3] = oldGrid(i)[3] + dt*deltaqi[1];
				
				//std::cout << qf_comp(i)[0] << ", " << qf_comp(i)[1] << "\n";
			}
			
			swap(oldGrid,newGrid);
		}
		
		//3d sim
		else if(dim == 3) {
			for (int i=0; i<nodes(oldGrid); ++i) {
				vector<int> x=position(oldGrid,i);
				
				// calculate grad(phi)
				vector<T> gradphi(dim,0.); // (0,0[,0])
				vector<vector<T> > gradq(2*dim-2, gradphi);
				for (int d=0; d<dim; ++d) {
					++x[d];
					T phir=oldGrid(x)[0];
					vector<T> qr(2*dim-2,0.);
					for(int j = 0; j < 2*dim-2; j++) {
						qr[j] = oldGrid(x)[j+2];
					}
					--x[d];
					gradphi[d]=(phir-oldGrid(x)[0])/deltax;
					for(int j = 0; j < 2*dim-2; j++) {
						gradq[j][d] = (qr[j]-oldGrid(x)[2+j])/deltax;
					}
				}
				dphi(i) = gradphi;
				dq(i) = gradq;
			
			}
			ghostswap(dphi);
			ghostswap(dq);
			
			for (int i=0; i<nodes(oldGrid); ++i) {
				vector<int> x=position(oldGrid,i);
				vector<T> q_av(4, 0.);
				vector<T> dphi_av(3, 0);
				dphi_av[0] += dphi(x)[0];
				dphi_av[1] += dphi(x)[1];
				dphi_av[2] += dphi(x)[2];
				for(int j = 0; j < 4; j++) {
					q_av[j] += oldGrid(x)[2+j];
				}
				++x[0];
				dphi_av[1] += dphi(x)[1];
				dphi_av[2] += dphi(x)[2];
				for(int j = 0; j < 4; j++) {
					q_av[j] += oldGrid(x)[2+j];
				}
				++x[1];
				dphi_av[2] += dphi(x)[2];
				for(int j = 0; j < 4; j++) {
					q_av[j] += oldGrid(x)[2+j];
				}
				--x[0];
				dphi_av[0] += dphi(x)[0];
				dphi_av[2] += dphi(x)[2];
				for(int j = 0; j < 4; j++) {
					q_av[j] += oldGrid(x)[2+j];
				}
				++x[2];
				dphi_av[0] += dphi(x)[0];
				for(int j = 0; j < 4; j++) {
					q_av[j] += oldGrid(x)[2+j];
				}
				++x[0];
				for(int j = 0; j < 4; j++) {
					q_av[j] += oldGrid(x)[2+j];
				}
				--x[1];
				dphi_av[1] += dphi(x)[1];
				for(int j = 0; j < 4; j++) {
					q_av[j] += oldGrid(x)[2+j];
				}
				--x[0];
				dphi_av[0] += dphi(x)[0];
				dphi_av[1] += dphi(x)[1];
				for(int j = 0; j < 3; j++) {
					dphi_av[j] /= 4;
				}
				for(int j = 0; j < 4; j++) {
					q_av[j] += oldGrid(x)[2+j];
					q_av[j] /= 8;
				}
				vq(i) = q_av;
				vdphi(i) = dphi_av;
			}
			for (int i=0; i<nodes(oldGrid); ++i) {
				g(i) = oldGrid(i)[0]*oldGrid(i)[0]*(1-oldGrid(i)[0])*(1-oldGrid(i)[0]);
				gprime(i) = 4*oldGrid(i)[0]*oldGrid(i)[0]*oldGrid(i)[0]-6*oldGrid(i)[0]*oldGrid(i)[0]+2*oldGrid(i)[0];
				D_C(i) = D_L + (D_S-D_L)*oldGrid(i)[0]*oldGrid(i)[0]*oldGrid(i)[0]*(10-15*oldGrid(i)[0]+6*oldGrid(i)[0]*oldGrid(i)[0]);
				Q(i) = D_C(i)*v_m*oldGrid(i)[1]*(1-oldGrid(i)[1])*(gprime(i)*(W_B-W_A)+30*g(i)*(L_B*(1/T_mB-1/temp) - L_A*(1/T_mA-1/temp)))/R;
			
				vdpsi(i)[0] = (vq(i)[0]*vq(i)[0]+vq(i)[1]*vq(i)[1]-vq(i)[2]*vq(i)[2]-vq(i)[3]*vq(i)[3])*vdphi(i)[0];
				vdpsi(i)[0] += (2*vq(i)[1]*vq(i)[2]-2*vq(i)[0]*vq(i)[3])*vdphi(i)[1];
				vdpsi(i)[0] += (2*vq(i)[1]*vq(i)[3]+2*vq(i)[0]*vq(i)[2])*vdphi(i)[2];
				vdpsi(i)[1] = (2*vq(i)[1]*vq(i)[2]+2*vq(i)[0]*vq(i)[3])*vdphi(i)[0];
				vdpsi(i)[1] += (vq(i)[0]*vq(i)[0]-vq(i)[1]*vq(i)[1]+vq(i)[2]*vq(i)[2]-vq(i)[3]*vq(i)[3])*vdphi(i)[1];
				vdpsi(i)[1] += (2*vq(i)[2]*vq(i)[3]-2*vq(i)[0]*vq(i)[1])*vdphi(i)[2];
				vdpsi(i)[2] = (2*vq(i)[1]*vq(i)[3]-2*vq(i)[0]*vq(i)[2])*vdphi(i)[0];
				vdpsi(i)[2] += (2*vq(i)[2]*vq(i)[3]+2*vq(i)[0]*vq(i)[1])*vdphi(i)[1];
				vdpsi(i)[2] += (vq(i)[0]*vq(i)[0]-vq(i)[1]*vq(i)[1]-vq(i)[2]*vq(i)[2]+vq(i)[3]*vq(i)[3])*vdphi(i)[2];
				vmgphi2(i) = vdphi(i)[0]*vdphi(i)[0]+vdphi(i)[1]*vdphi(i)[1]+vdphi(i)[2]*vdphi(i)[2]+0.000000000001; //same as above
			}
			
			ghostswap(vq);
			ghostswap(vdphi);
			ghostswap(vdpsi);
			ghostswap(vmgphi2);
			ghostswap(D_C);
			ghostswap(Q);
			
			for (int i=0; i<nodes(oldGrid); ++i) {
				vector<int> x=position(oldGrid,i);
				vector<T> dpsi_av(3, 0.);
				vector<T> pfcomp_av(3, 0.);
				vector<T> dqs_av(3, 0.);
				vector<T> qfcomp_av(4, 0.);
				T _lphi = (dphi(x)[0] + dphi(x)[1] + dphi(x)[2])*(1-3*y_e);
				T psix3 = vdpsi(x)[0]*vdpsi(x)[0]*vdpsi(x)[0];
				T psiy3 = vdpsi(x)[1]*vdpsi(x)[1]*vdpsi(x)[1];
				T psiz3 = vdpsi(x)[2]*vdpsi(x)[2]*vdpsi(x)[2];
				T psi4mgphi4 = (psix3*vdpsi(x)[0] + psiy3*vdpsi(x)[1] + psiz3*vdpsi(x)[2])/(vmgphi2(x)*vmgphi2(x));
				qfcomp_av[0] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[0]*vdphi(x)[0] - vq(x)[3]*vdphi(x)[1] + vq(x)[2]*vdphi(x)[2])+psiy3*(vq(x)[0]*vdphi(x)[1] + vq(x)[1]*vdphi(x)[0] - vq(x)[1]*vdphi(x)[2]) + psiz3*(vq(x)[1]*vdphi(x)[1] + vq(x)[0]*vdphi(x)[2] - vq(x)[2]*vdphi(x)[0]));
				qfcomp_av[1] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[1]*vdphi(x)[0] + vq(x)[2]*vdphi(x)[1] + vq(x)[3]*vdphi(x)[2])+psiy3*(vq(x)[2]*vdphi(x)[0] - vq(x)[1]*vdphi(x)[1] - vq(x)[0]*vdphi(x)[2]) + psiz3*(vq(x)[3]*vdphi(x)[0] + vq(x)[0]*vdphi(x)[1] - vq(x)[1]*vdphi(x)[2]));
				qfcomp_av[2] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[1]*vdphi(x)[1] - vq(x)[2]*vdphi(x)[0] + vq(x)[0]*vdphi(x)[2])+psiy3*(vq(x)[1]*vdphi(x)[0] + vq(x)[2]*vdphi(x)[1] + vq(x)[3]*vdphi(x)[2]) + psiz3*(vq(x)[3]*vdphi(x)[1] - vq(x)[0]*vdphi(x)[0] - vq(x)[2]*vdphi(x)[2]));
				qfcomp_av[3] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[1]*vdphi(x)[2] - vq(x)[0]*vdphi(x)[1] - vq(x)[3]*vdphi(x)[0])+psiy3*(vq(x)[0]*vdphi(x)[0] - vq(x)[3]*vdphi(x)[1] + vq(x)[2]*vdphi(x)[2]) + psiz3*(vq(x)[1]*vdphi(x)[0] + vq(x)[2]*vdphi(x)[1] - vq(x)[3]*vdphi(x)[2]));
				pfcomp_av[0] += 4*y_e*((2*psix3*(vq(x)[0]*vq(x)[0]+vq(x)[1]*vq(x)[1]-vq(x)[2]*vq(x)[2]-vq(x)[3]*vq(x)[3]) + 4*psiy3*(vq(x)[0]*vq(x)[3]+vq(x)[1]*vq(x)[2]) + 4*psiz3*(vq(x)[1]*vq(x)[3]-vq(x)[0]*vq(x)[2]))/vmgphi2(x) - vdphi(x)[0]*psi4mgphi4);
				pfcomp_av[1] += 4*y_e*((4*psix3*(vq(x)[1]*vq(x)[2]-vq(x)[0]*vq(x)[3]) + 2*psiy3*(vq(x)[0]*vq(x)[0]-vq(x)[1]*vq(x)[1]+vq(x)[2]*vq(x)[2]-vq(x)[3]*vq(x)[3]) + 4*psiz3*(vq(x)[2]*vq(x)[3]+vq(x)[0]*vq(x)[1]))/vmgphi2(x) - vdphi(x)[1]*psi4mgphi4);
				pfcomp_av[2] += 4*y_e*((4*psix3*(vq(x)[1]*vq(x)[3]+vq(x)[0]*vq(x)[2]) + 4*psiy3*(vq(x)[2]*vq(x)[3]-vq(x)[0]*vq(x)[1]) + 2*psiz3*(vq(x)[0]*vq(x)[0]-vq(x)[1]*vq(x)[1]-vq(x)[2]*vq(x)[2]+vq(x)[3]*vq(x)[3]))/vmgphi2(x) - vdphi(x)[2]*psi4mgphi4);
				for(int d = 0; d < 3; d++) {
					dpsi_av[d] += vdpsi(x)[d];
					for(int j = 0; j < 4; j++) {
						dqs_av[d] = dq(x)[j][d]*dq(x)[j][d];
					}
				}
				--x[0];
				_lphi -= (1-3*y_e)*dphi(x)[0];
				for(int j = 0; j < 4; j++) {
					dqs_av[0] += dq(x)[j][0]*dq(x)[j][0];
				}
				psix3 = vdpsi(x)[0]*vdpsi(x)[0]*vdpsi(x)[0];
				psiy3 = vdpsi(x)[1]*vdpsi(x)[1]*vdpsi(x)[1];
				psiz3 = vdpsi(x)[2]*vdpsi(x)[2]*vdpsi(x)[2];
				psi4mgphi4 = (psix3*vdpsi(x)[0] + psiy3*vdpsi(x)[1] + psiz3*vdpsi(x)[2])/(vmgphi2(x)*vmgphi2(x));
				qfcomp_av[0] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[0]*vdphi(x)[0] - vq(x)[3]*vdphi(x)[1] + vq(x)[2]*vdphi(x)[2])+psiy3*(vq(x)[0]*vdphi(x)[1] + vq(x)[1]*vdphi(x)[0] - vq(x)[1]*vdphi(x)[2]) + psiz3*(vq(x)[1]*vdphi(x)[1] + vq(x)[0]*vdphi(x)[2] - vq(x)[2]*vdphi(x)[0]));
				qfcomp_av[1] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[1]*vdphi(x)[0] + vq(x)[2]*vdphi(x)[1] + vq(x)[3]*vdphi(x)[2])+psiy3*(vq(x)[2]*vdphi(x)[0] - vq(x)[1]*vdphi(x)[1] - vq(x)[0]*vdphi(x)[2]) + psiz3*(vq(x)[3]*vdphi(x)[0] + vq(x)[0]*vdphi(x)[1] - vq(x)[1]*vdphi(x)[2]));
				qfcomp_av[2] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[1]*vdphi(x)[1] - vq(x)[2]*vdphi(x)[0] + vq(x)[0]*vdphi(x)[2])+psiy3*(vq(x)[1]*vdphi(x)[0] + vq(x)[2]*vdphi(x)[1] + vq(x)[3]*vdphi(x)[2]) + psiz3*(vq(x)[3]*vdphi(x)[1] - vq(x)[0]*vdphi(x)[0] - vq(x)[2]*vdphi(x)[2]));
				qfcomp_av[3] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[1]*vdphi(x)[2] - vq(x)[0]*vdphi(x)[1] - vq(x)[3]*vdphi(x)[0])+psiy3*(vq(x)[0]*vdphi(x)[0] - vq(x)[3]*vdphi(x)[1] + vq(x)[2]*vdphi(x)[2]) + psiz3*(vq(x)[1]*vdphi(x)[0] + vq(x)[2]*vdphi(x)[1] - vq(x)[3]*vdphi(x)[2]));
				pfcomp_av[0] -= 4*y_e*((2*psix3*(vq(x)[0]*vq(x)[0]+vq(x)[1]*vq(x)[1]-vq(x)[2]*vq(x)[2]-vq(x)[3]*vq(x)[3]) + 4*psiy3*(vq(x)[0]*vq(x)[3]+vq(x)[1]*vq(x)[2]) + 4*psiz3*(vq(x)[1]*vq(x)[3]-vq(x)[0]*vq(x)[2]))/vmgphi2(x) - vdphi(x)[0]*psi4mgphi4);
				pfcomp_av[1] += 4*y_e*((4*psix3*(vq(x)[1]*vq(x)[2]-vq(x)[0]*vq(x)[3]) + 2*psiy3*(vq(x)[0]*vq(x)[0]-vq(x)[1]*vq(x)[1]+vq(x)[2]*vq(x)[2]-vq(x)[3]*vq(x)[3]) + 4*psiz3*(vq(x)[2]*vq(x)[3]+vq(x)[0]*vq(x)[1]))/vmgphi2(x) - vdphi(x)[1]*psi4mgphi4);
				pfcomp_av[2] += 4*y_e*((4*psix3*(vq(x)[1]*vq(x)[3]+vq(x)[0]*vq(x)[2]) + 4*psiy3*(vq(x)[2]*vq(x)[3]-vq(x)[0]*vq(x)[1]) + 2*psiz3*(vq(x)[0]*vq(x)[0]-vq(x)[1]*vq(x)[1]-vq(x)[2]*vq(x)[2]+vq(x)[3]*vq(x)[3]))/vmgphi2(x) - vdphi(x)[2]*psi4mgphi4);
				for(int j = 0; j < 3; j++) {
					dpsi_av[j] += vdpsi(x)[j];
				}
				--x[1];
				psix3 = vdpsi(x)[0]*vdpsi(x)[0]*vdpsi(x)[0];
				psiy3 = vdpsi(x)[1]*vdpsi(x)[1]*vdpsi(x)[1];
				psiz3 = vdpsi(x)[2]*vdpsi(x)[2]*vdpsi(x)[2];
				psi4mgphi4 = (psix3*vdpsi(x)[0] + psiy3*vdpsi(x)[1] + psiz3*vdpsi(x)[2])/(vmgphi2(x)*vmgphi2(x));
				qfcomp_av[0] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[0]*vdphi(x)[0] - vq(x)[3]*vdphi(x)[1] + vq(x)[2]*vdphi(x)[2])+psiy3*(vq(x)[0]*vdphi(x)[1] + vq(x)[1]*vdphi(x)[0] - vq(x)[1]*vdphi(x)[2]) + psiz3*(vq(x)[1]*vdphi(x)[1] + vq(x)[0]*vdphi(x)[2] - vq(x)[2]*vdphi(x)[0]));
				qfcomp_av[1] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[1]*vdphi(x)[0] + vq(x)[2]*vdphi(x)[1] + vq(x)[3]*vdphi(x)[2])+psiy3*(vq(x)[2]*vdphi(x)[0] - vq(x)[1]*vdphi(x)[1] - vq(x)[0]*vdphi(x)[2]) + psiz3*(vq(x)[3]*vdphi(x)[0] + vq(x)[0]*vdphi(x)[1] - vq(x)[1]*vdphi(x)[2]));
				qfcomp_av[2] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[1]*vdphi(x)[1] - vq(x)[2]*vdphi(x)[0] + vq(x)[0]*vdphi(x)[2])+psiy3*(vq(x)[1]*vdphi(x)[0] + vq(x)[2]*vdphi(x)[1] + vq(x)[3]*vdphi(x)[2]) + psiz3*(vq(x)[3]*vdphi(x)[1] - vq(x)[0]*vdphi(x)[0] - vq(x)[2]*vdphi(x)[2]));
				qfcomp_av[3] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[1]*vdphi(x)[2] - vq(x)[0]*vdphi(x)[1] - vq(x)[3]*vdphi(x)[0])+psiy3*(vq(x)[0]*vdphi(x)[0] - vq(x)[3]*vdphi(x)[1] + vq(x)[2]*vdphi(x)[2]) + psiz3*(vq(x)[1]*vdphi(x)[0] + vq(x)[2]*vdphi(x)[1] - vq(x)[3]*vdphi(x)[2]));
				pfcomp_av[0] -= 4*y_e*((2*psix3*(vq(x)[0]*vq(x)[0]+vq(x)[1]*vq(x)[1]-vq(x)[2]*vq(x)[2]-vq(x)[3]*vq(x)[3]) + 4*psiy3*(vq(x)[0]*vq(x)[3]+vq(x)[1]*vq(x)[2]) + 4*psiz3*(vq(x)[1]*vq(x)[3]-vq(x)[0]*vq(x)[2]))/vmgphi2(x) - vdphi(x)[0]*psi4mgphi4);
				pfcomp_av[1] -= 4*y_e*((4*psix3*(vq(x)[1]*vq(x)[2]-vq(x)[0]*vq(x)[3]) + 2*psiy3*(vq(x)[0]*vq(x)[0]-vq(x)[1]*vq(x)[1]+vq(x)[2]*vq(x)[2]-vq(x)[3]*vq(x)[3]) + 4*psiz3*(vq(x)[2]*vq(x)[3]+vq(x)[0]*vq(x)[1]))/vmgphi2(x) - vdphi(x)[1]*psi4mgphi4);
				pfcomp_av[2] += 4*y_e*((4*psix3*(vq(x)[1]*vq(x)[3]+vq(x)[0]*vq(x)[2]) + 4*psiy3*(vq(x)[2]*vq(x)[3]-vq(x)[0]*vq(x)[1]) + 2*psiz3*(vq(x)[0]*vq(x)[0]-vq(x)[1]*vq(x)[1]-vq(x)[2]*vq(x)[2]+vq(x)[3]*vq(x)[3]))/vmgphi2(x) - vdphi(x)[2]*psi4mgphi4);
				for(int j = 0; j < 3; j++) {
					dpsi_av[j] += vdpsi(x)[j];
				}
				++x[0];
				_lphi -= (1-3*y_e)*dphi(x)[1];
				for(int j = 0; j < 4; j++) {
					dqs_av[1] += dq(x)[j][1]*dq(x)[j][1];
				}
				psix3 = vdpsi(x)[0]*vdpsi(x)[0]*vdpsi(x)[0];
				psiy3 = vdpsi(x)[1]*vdpsi(x)[1]*vdpsi(x)[1];
				psiz3 = vdpsi(x)[2]*vdpsi(x)[2]*vdpsi(x)[2];
				psi4mgphi4 = (psix3*vdpsi(x)[0] + psiy3*vdpsi(x)[1] + psiz3*vdpsi(x)[2])/(vmgphi2(x)*vmgphi2(x));
				qfcomp_av[0] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[0]*vdphi(x)[0] - vq(x)[3]*vdphi(x)[1] + vq(x)[2]*vdphi(x)[2])+psiy3*(vq(x)[0]*vdphi(x)[1] + vq(x)[1]*vdphi(x)[0] - vq(x)[1]*vdphi(x)[2]) + psiz3*(vq(x)[1]*vdphi(x)[1] + vq(x)[0]*vdphi(x)[2] - vq(x)[2]*vdphi(x)[0]));
				qfcomp_av[1] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[1]*vdphi(x)[0] + vq(x)[2]*vdphi(x)[1] + vq(x)[3]*vdphi(x)[2])+psiy3*(vq(x)[2]*vdphi(x)[0] - vq(x)[1]*vdphi(x)[1] - vq(x)[0]*vdphi(x)[2]) + psiz3*(vq(x)[3]*vdphi(x)[0] + vq(x)[0]*vdphi(x)[1] - vq(x)[1]*vdphi(x)[2]));
				qfcomp_av[2] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[1]*vdphi(x)[1] - vq(x)[2]*vdphi(x)[0] + vq(x)[0]*vdphi(x)[2])+psiy3*(vq(x)[1]*vdphi(x)[0] + vq(x)[2]*vdphi(x)[1] + vq(x)[3]*vdphi(x)[2]) + psiz3*(vq(x)[3]*vdphi(x)[1] - vq(x)[0]*vdphi(x)[0] - vq(x)[2]*vdphi(x)[2]));
				qfcomp_av[3] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[1]*vdphi(x)[2] - vq(x)[0]*vdphi(x)[1] - vq(x)[3]*vdphi(x)[0])+psiy3*(vq(x)[0]*vdphi(x)[0] - vq(x)[3]*vdphi(x)[1] + vq(x)[2]*vdphi(x)[2]) + psiz3*(vq(x)[1]*vdphi(x)[0] + vq(x)[2]*vdphi(x)[1] - vq(x)[3]*vdphi(x)[2]));
				pfcomp_av[0] += 4*y_e*((2*psix3*(vq(x)[0]*vq(x)[0]+vq(x)[1]*vq(x)[1]-vq(x)[2]*vq(x)[2]-vq(x)[3]*vq(x)[3]) + 4*psiy3*(vq(x)[0]*vq(x)[3]+vq(x)[1]*vq(x)[2]) + 4*psiz3*(vq(x)[1]*vq(x)[3]-vq(x)[0]*vq(x)[2]))/vmgphi2(x) - vdphi(x)[0]*psi4mgphi4);
				pfcomp_av[1] -= 4*y_e*((4*psix3*(vq(x)[1]*vq(x)[2]-vq(x)[0]*vq(x)[3]) + 2*psiy3*(vq(x)[0]*vq(x)[0]-vq(x)[1]*vq(x)[1]+vq(x)[2]*vq(x)[2]-vq(x)[3]*vq(x)[3]) + 4*psiz3*(vq(x)[2]*vq(x)[3]+vq(x)[0]*vq(x)[1]))/vmgphi2(x) - vdphi(x)[1]*psi4mgphi4);
				pfcomp_av[2] += 4*y_e*((4*psix3*(vq(x)[1]*vq(x)[3]+vq(x)[0]*vq(x)[2]) + 4*psiy3*(vq(x)[2]*vq(x)[3]-vq(x)[0]*vq(x)[1]) + 2*psiz3*(vq(x)[0]*vq(x)[0]-vq(x)[1]*vq(x)[1]-vq(x)[2]*vq(x)[2]+vq(x)[3]*vq(x)[3]))/vmgphi2(x) - vdphi(x)[2]*psi4mgphi4);
				for(int j = 0; j < 3; j++) {
					dpsi_av[j] += vdpsi(x)[j];
				}
				--x[2];
				psix3 = vdpsi(x)[0]*vdpsi(x)[0]*vdpsi(x)[0];
				psiy3 = vdpsi(x)[1]*vdpsi(x)[1]*vdpsi(x)[1];
				psiz3 = vdpsi(x)[2]*vdpsi(x)[2]*vdpsi(x)[2];
				psi4mgphi4 = (psix3*vdpsi(x)[0] + psiy3*vdpsi(x)[1] + psiz3*vdpsi(x)[2])/(vmgphi2(x)*vmgphi2(x));
				qfcomp_av[0] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[0]*vdphi(x)[0] - vq(x)[3]*vdphi(x)[1] + vq(x)[2]*vdphi(x)[2])+psiy3*(vq(x)[0]*vdphi(x)[1] + vq(x)[1]*vdphi(x)[0] - vq(x)[1]*vdphi(x)[2]) + psiz3*(vq(x)[1]*vdphi(x)[1] + vq(x)[0]*vdphi(x)[2] - vq(x)[2]*vdphi(x)[0]));
				qfcomp_av[1] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[1]*vdphi(x)[0] + vq(x)[2]*vdphi(x)[1] + vq(x)[3]*vdphi(x)[2])+psiy3*(vq(x)[2]*vdphi(x)[0] - vq(x)[1]*vdphi(x)[1] - vq(x)[0]*vdphi(x)[2]) + psiz3*(vq(x)[3]*vdphi(x)[0] + vq(x)[0]*vdphi(x)[1] - vq(x)[1]*vdphi(x)[2]));
				qfcomp_av[2] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[1]*vdphi(x)[1] - vq(x)[2]*vdphi(x)[0] + vq(x)[0]*vdphi(x)[2])+psiy3*(vq(x)[1]*vdphi(x)[0] + vq(x)[2]*vdphi(x)[1] + vq(x)[3]*vdphi(x)[2]) + psiz3*(vq(x)[3]*vdphi(x)[1] - vq(x)[0]*vdphi(x)[0] - vq(x)[2]*vdphi(x)[2]));
				qfcomp_av[3] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[1]*vdphi(x)[2] - vq(x)[0]*vdphi(x)[1] - vq(x)[3]*vdphi(x)[0])+psiy3*(vq(x)[0]*vdphi(x)[0] - vq(x)[3]*vdphi(x)[1] + vq(x)[2]*vdphi(x)[2]) + psiz3*(vq(x)[1]*vdphi(x)[0] + vq(x)[2]*vdphi(x)[1] - vq(x)[3]*vdphi(x)[2]));
				pfcomp_av[0] += 4*y_e*((2*psix3*(vq(x)[0]*vq(x)[0]+vq(x)[1]*vq(x)[1]-vq(x)[2]*vq(x)[2]-vq(x)[3]*vq(x)[3]) + 4*psiy3*(vq(x)[0]*vq(x)[3]+vq(x)[1]*vq(x)[2]) + 4*psiz3*(vq(x)[1]*vq(x)[3]-vq(x)[0]*vq(x)[2]))/vmgphi2(x) - vdphi(x)[0]*psi4mgphi4);
				pfcomp_av[1] -= 4*y_e*((4*psix3*(vq(x)[1]*vq(x)[2]-vq(x)[0]*vq(x)[3]) + 2*psiy3*(vq(x)[0]*vq(x)[0]-vq(x)[1]*vq(x)[1]+vq(x)[2]*vq(x)[2]-vq(x)[3]*vq(x)[3]) + 4*psiz3*(vq(x)[2]*vq(x)[3]+vq(x)[0]*vq(x)[1]))/vmgphi2(x) - vdphi(x)[1]*psi4mgphi4);
				pfcomp_av[2] -= 4*y_e*((4*psix3*(vq(x)[1]*vq(x)[3]+vq(x)[0]*vq(x)[2]) + 4*psiy3*(vq(x)[2]*vq(x)[3]-vq(x)[0]*vq(x)[1]) + 2*psiz3*(vq(x)[0]*vq(x)[0]-vq(x)[1]*vq(x)[1]-vq(x)[2]*vq(x)[2]+vq(x)[3]*vq(x)[3]))/vmgphi2(x) - vdphi(x)[2]*psi4mgphi4);
				for(int j = 0; j < 3; j++) {
					dpsi_av[j] += vdpsi(x)[j];
				}
				--x[0];
				psix3 = vdpsi(x)[0]*vdpsi(x)[0]*vdpsi(x)[0];
				psiy3 = vdpsi(x)[1]*vdpsi(x)[1]*vdpsi(x)[1];
				psiz3 = vdpsi(x)[2]*vdpsi(x)[2]*vdpsi(x)[2];
				psi4mgphi4 = (psix3*vdpsi(x)[0] + psiy3*vdpsi(x)[1] + psiz3*vdpsi(x)[2])/(vmgphi2(x)*vmgphi2(x));
				qfcomp_av[0] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[0]*vdphi(x)[0] - vq(x)[3]*vdphi(x)[1] + vq(x)[2]*vdphi(x)[2])+psiy3*(vq(x)[0]*vdphi(x)[1] + vq(x)[1]*vdphi(x)[0] - vq(x)[1]*vdphi(x)[2]) + psiz3*(vq(x)[1]*vdphi(x)[1] + vq(x)[0]*vdphi(x)[2] - vq(x)[2]*vdphi(x)[0]));
				qfcomp_av[1] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[1]*vdphi(x)[0] + vq(x)[2]*vdphi(x)[1] + vq(x)[3]*vdphi(x)[2])+psiy3*(vq(x)[2]*vdphi(x)[0] - vq(x)[1]*vdphi(x)[1] - vq(x)[0]*vdphi(x)[2]) + psiz3*(vq(x)[3]*vdphi(x)[0] + vq(x)[0]*vdphi(x)[1] - vq(x)[1]*vdphi(x)[2]));
				qfcomp_av[2] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[1]*vdphi(x)[1] - vq(x)[2]*vdphi(x)[0] + vq(x)[0]*vdphi(x)[2])+psiy3*(vq(x)[1]*vdphi(x)[0] + vq(x)[2]*vdphi(x)[1] + vq(x)[3]*vdphi(x)[2]) + psiz3*(vq(x)[3]*vdphi(x)[1] - vq(x)[0]*vdphi(x)[0] - vq(x)[2]*vdphi(x)[2]));
				qfcomp_av[3] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[1]*vdphi(x)[2] - vq(x)[0]*vdphi(x)[1] - vq(x)[3]*vdphi(x)[0])+psiy3*(vq(x)[0]*vdphi(x)[0] - vq(x)[3]*vdphi(x)[1] + vq(x)[2]*vdphi(x)[2]) + psiz3*(vq(x)[1]*vdphi(x)[0] + vq(x)[2]*vdphi(x)[1] - vq(x)[3]*vdphi(x)[2]));
				pfcomp_av[0] -= 4*y_e*((2*psix3*(vq(x)[0]*vq(x)[0]+vq(x)[1]*vq(x)[1]-vq(x)[2]*vq(x)[2]-vq(x)[3]*vq(x)[3]) + 4*psiy3*(vq(x)[0]*vq(x)[3]+vq(x)[1]*vq(x)[2]) + 4*psiz3*(vq(x)[1]*vq(x)[3]-vq(x)[0]*vq(x)[2]))/vmgphi2(x) - vdphi(x)[0]*psi4mgphi4);
				pfcomp_av[1] -= 4*y_e*((4*psix3*(vq(x)[1]*vq(x)[2]-vq(x)[0]*vq(x)[3]) + 2*psiy3*(vq(x)[0]*vq(x)[0]-vq(x)[1]*vq(x)[1]+vq(x)[2]*vq(x)[2]-vq(x)[3]*vq(x)[3]) + 4*psiz3*(vq(x)[2]*vq(x)[3]+vq(x)[0]*vq(x)[1]))/vmgphi2(x) - vdphi(x)[1]*psi4mgphi4);
				pfcomp_av[2] -= 4*y_e*((4*psix3*(vq(x)[1]*vq(x)[3]+vq(x)[0]*vq(x)[2]) + 4*psiy3*(vq(x)[2]*vq(x)[3]-vq(x)[0]*vq(x)[1]) + 2*psiz3*(vq(x)[0]*vq(x)[0]-vq(x)[1]*vq(x)[1]-vq(x)[2]*vq(x)[2]+vq(x)[3]*vq(x)[3]))/vmgphi2(x) - vdphi(x)[2]*psi4mgphi4);
				for(int j = 0; j < 3; j++) {
					dpsi_av[j] += vdpsi(x)[j];
				}
				++x[1];
				psix3 = vdpsi(x)[0]*vdpsi(x)[0]*vdpsi(x)[0];
				psiy3 = vdpsi(x)[1]*vdpsi(x)[1]*vdpsi(x)[1];
				psiz3 = vdpsi(x)[2]*vdpsi(x)[2]*vdpsi(x)[2];
				psi4mgphi4 = (psix3*vdpsi(x)[0] + psiy3*vdpsi(x)[1] + psiz3*vdpsi(x)[2])/(vmgphi2(x)*vmgphi2(x));
				qfcomp_av[0] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[0]*vdphi(x)[0] - vq(x)[3]*vdphi(x)[1] + vq(x)[2]*vdphi(x)[2])+psiy3*(vq(x)[0]*vdphi(x)[1] + vq(x)[1]*vdphi(x)[0] - vq(x)[1]*vdphi(x)[2]) + psiz3*(vq(x)[1]*vdphi(x)[1] + vq(x)[0]*vdphi(x)[2] - vq(x)[2]*vdphi(x)[0]));
				qfcomp_av[1] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[1]*vdphi(x)[0] + vq(x)[2]*vdphi(x)[1] + vq(x)[3]*vdphi(x)[2])+psiy3*(vq(x)[2]*vdphi(x)[0] - vq(x)[1]*vdphi(x)[1] - vq(x)[0]*vdphi(x)[2]) + psiz3*(vq(x)[3]*vdphi(x)[0] + vq(x)[0]*vdphi(x)[1] - vq(x)[1]*vdphi(x)[2]));
				qfcomp_av[2] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[1]*vdphi(x)[1] - vq(x)[2]*vdphi(x)[0] + vq(x)[0]*vdphi(x)[2])+psiy3*(vq(x)[1]*vdphi(x)[0] + vq(x)[2]*vdphi(x)[1] + vq(x)[3]*vdphi(x)[2]) + psiz3*(vq(x)[3]*vdphi(x)[1] - vq(x)[0]*vdphi(x)[0] - vq(x)[2]*vdphi(x)[2]));
				qfcomp_av[3] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[1]*vdphi(x)[2] - vq(x)[0]*vdphi(x)[1] - vq(x)[3]*vdphi(x)[0])+psiy3*(vq(x)[0]*vdphi(x)[0] - vq(x)[3]*vdphi(x)[1] + vq(x)[2]*vdphi(x)[2]) + psiz3*(vq(x)[1]*vdphi(x)[0] + vq(x)[2]*vdphi(x)[1] - vq(x)[3]*vdphi(x)[2]));
				pfcomp_av[0] -= 4*y_e*((2*psix3*(vq(x)[0]*vq(x)[0]+vq(x)[1]*vq(x)[1]-vq(x)[2]*vq(x)[2]-vq(x)[3]*vq(x)[3]) + 4*psiy3*(vq(x)[0]*vq(x)[3]+vq(x)[1]*vq(x)[2]) + 4*psiz3*(vq(x)[1]*vq(x)[3]-vq(x)[0]*vq(x)[2]))/vmgphi2(x) - vdphi(x)[0]*psi4mgphi4);
				pfcomp_av[1] += 4*y_e*((4*psix3*(vq(x)[1]*vq(x)[2]-vq(x)[0]*vq(x)[3]) + 2*psiy3*(vq(x)[0]*vq(x)[0]-vq(x)[1]*vq(x)[1]+vq(x)[2]*vq(x)[2]-vq(x)[3]*vq(x)[3]) + 4*psiz3*(vq(x)[2]*vq(x)[3]+vq(x)[0]*vq(x)[1]))/vmgphi2(x) - vdphi(x)[1]*psi4mgphi4);
				pfcomp_av[2] -= 4*y_e*((4*psix3*(vq(x)[1]*vq(x)[3]+vq(x)[0]*vq(x)[2]) + 4*psiy3*(vq(x)[2]*vq(x)[3]-vq(x)[0]*vq(x)[1]) + 2*psiz3*(vq(x)[0]*vq(x)[0]-vq(x)[1]*vq(x)[1]-vq(x)[2]*vq(x)[2]+vq(x)[3]*vq(x)[3]))/vmgphi2(x) - vdphi(x)[2]*psi4mgphi4);
				for(int j = 0; j < 3; j++) {
					dpsi_av[j] += vdpsi(x)[j];
				}
				++x[0];
				_lphi -= (1-3*y_e)*dphi(x)[2];
				for(int j = 0; j < 4; j++) {
					dqs_av[2] += dq(x)[j][2]*dq(x)[j][2];
				}
				psix3 = vdpsi(x)[0]*vdpsi(x)[0]*vdpsi(x)[0];
				psiy3 = vdpsi(x)[1]*vdpsi(x)[1]*vdpsi(x)[1];
				psiz3 = vdpsi(x)[2]*vdpsi(x)[2]*vdpsi(x)[2];
				psi4mgphi4 = (psix3*vdpsi(x)[0] + psiy3*vdpsi(x)[1] + psiz3*vdpsi(x)[2])/(vmgphi2(x)*vmgphi2(x));
				qfcomp_av[0] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[0]*vdphi(x)[0] - vq(x)[3]*vdphi(x)[1] + vq(x)[2]*vdphi(x)[2])+psiy3*(vq(x)[0]*vdphi(x)[1] + vq(x)[1]*vdphi(x)[0] - vq(x)[1]*vdphi(x)[2]) + psiz3*(vq(x)[1]*vdphi(x)[1] + vq(x)[0]*vdphi(x)[2] - vq(x)[2]*vdphi(x)[0]));
				qfcomp_av[1] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[1]*vdphi(x)[0] + vq(x)[2]*vdphi(x)[1] + vq(x)[3]*vdphi(x)[2])+psiy3*(vq(x)[2]*vdphi(x)[0] - vq(x)[1]*vdphi(x)[1] - vq(x)[0]*vdphi(x)[2]) + psiz3*(vq(x)[3]*vdphi(x)[0] + vq(x)[0]*vdphi(x)[1] - vq(x)[1]*vdphi(x)[2]));
				qfcomp_av[2] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[1]*vdphi(x)[1] - vq(x)[2]*vdphi(x)[0] + vq(x)[0]*vdphi(x)[2])+psiy3*(vq(x)[1]*vdphi(x)[0] + vq(x)[2]*vdphi(x)[1] + vq(x)[3]*vdphi(x)[2]) + psiz3*(vq(x)[3]*vdphi(x)[1] - vq(x)[0]*vdphi(x)[0] - vq(x)[2]*vdphi(x)[2]));
				qfcomp_av[3] += 16*ebar2*y_e/vmgphi2(x)*(psix3*(vq(x)[1]*vdphi(x)[2] - vq(x)[0]*vdphi(x)[1] - vq(x)[3]*vdphi(x)[0])+psiy3*(vq(x)[0]*vdphi(x)[0] - vq(x)[3]*vdphi(x)[1] + vq(x)[2]*vdphi(x)[2]) + psiz3*(vq(x)[1]*vdphi(x)[0] + vq(x)[2]*vdphi(x)[1] - vq(x)[3]*vdphi(x)[2]));
				pfcomp_av[0] += 4*y_e*((2*psix3*(vq(x)[0]*vq(x)[0]+vq(x)[1]*vq(x)[1]-vq(x)[2]*vq(x)[2]-vq(x)[3]*vq(x)[3]) + 4*psiy3*(vq(x)[0]*vq(x)[3]+vq(x)[1]*vq(x)[2]) + 4*psiz3*(vq(x)[1]*vq(x)[3]-vq(x)[0]*vq(x)[2]))/vmgphi2(x) - vdphi(x)[0]*psi4mgphi4);
				pfcomp_av[1] += 4*y_e*((4*psix3*(vq(x)[1]*vq(x)[2]-vq(x)[0]*vq(x)[3]) + 2*psiy3*(vq(x)[0]*vq(x)[0]-vq(x)[1]*vq(x)[1]+vq(x)[2]*vq(x)[2]-vq(x)[3]*vq(x)[3]) + 4*psiz3*(vq(x)[2]*vq(x)[3]+vq(x)[0]*vq(x)[1]))/vmgphi2(x) - vdphi(x)[1]*psi4mgphi4);
				pfcomp_av[2] -= 4*y_e*((4*psix3*(vq(x)[1]*vq(x)[3]+vq(x)[0]*vq(x)[2]) + 4*psiy3*(vq(x)[2]*vq(x)[3]-vq(x)[0]*vq(x)[1]) + 2*psiz3*(vq(x)[0]*vq(x)[0]-vq(x)[1]*vq(x)[1]-vq(x)[2]*vq(x)[2]+vq(x)[3]*vq(x)[3]))/vmgphi2(x) - vdphi(x)[2]*psi4mgphi4);
				
				for(int j = 0; j < 3; j++) {
					dpsi_av[j] += vdpsi(x)[j];
					dpsi_av[j] /= 8;
					pfcomp_av[j] /= (4.*deltax);
					qfcomp_av[j] /= 8;
				}
				lphi(i) = _lphi/deltax;
				qfcomp_av[3] /= 8;
				mgq(i) = sqrt(0.5*(dqs_av[0] + dqs_av[1] + dqs_av[2]));
				dpsi(i) = dpsi_av;
				pf_comp(i) = pfcomp_av;
			}
			
			for (int i=0; i<nodes(oldGrid); ++i) {
				newGrid(i)[0] = oldGrid(i)[0] - dt*((1-oldGrid(i)[1])*M_A+oldGrid(i)[1]*M_B)*((1-oldGrid(i)[1])*(W_A*gprime(i)+30*(1/T_mA - 1/temp)*g(i)*L_A)+oldGrid(i)[1]*(W_B*gprime(i)+30*(1/T_mB - 1/temp)*g(i)*L_B) + 4*H*temp*oldGrid(i)[0]*mgq(i) - ebar2*(lphi(i)+pf_comp(i)[0]+pf_comp(i)[1]+pf_comp(i)[2]));
			}
			
			for (int i=0; i<nodes(oldGrid); ++i) {
				vector<int> x=position(oldGrid,i);
				T deltac = 0.;
				for(int d = 0; d < dim; d++) {
					--x[d];
					T phi_l = oldGrid(x)[0];
					T c_l = oldGrid(x)[1];
					T dc_l = D_C(x);
					T Q_l = Q(x);
					x[d] += 2;
					T phi_r = oldGrid(x)[0];
					T c_r = oldGrid(x)[1];
					T dc_r = D_C(x);
					T Q_r = Q(x);
					--x[d];
					deltac += 0.5*((dc_r + D_C(x))*(c_r-oldGrid(x)[1]) - (dc_l + D_C(x))*(oldGrid(x)[1]-c_l))/(deltax*deltax) + 0.5*((Q_r + Q(x))*(phi_r-oldGrid(x)[0]) - (Q_l + Q(x))*(oldGrid(x)[0]-phi_l))/(deltax*deltax);
				}
				newGrid(i)[1] = oldGrid(i)[1] + dt*deltac;
			}
			
			for (int i=0; i<nodes(oldGrid); ++i) {
				vector<int> x=position(oldGrid,i);
				vector<T> dq1_l(3, 0.); //"left" gradient of q1, index is x,y
				vector<T> dq2_l(3, 0.);
				vector<T> dq3_l(3, 0.);
				vector<T> dq4_l(3, 0.);
				vector<T> dqs_cc(3, 0.); //cell-centered gradients squared, summed. index is in which dimension (x,y,z)
				vector<T> phi_l(3, 0.); //left value of phi
				vector<T> phi_r(3, 0.); //right value of phi
				vector<T> dphi_cc(3, 0.); //cell-centered gradient of phi
				vector<T> lq(4, 0.); //laplacian of q, index represents the quaternion index
				vector<T> rgqs_l(3, 0.); //the "left" root-grad-q-squared. Index says which direction (x,y) uses face-centered gradient to compute it
				vector<T> rgqs_r(3, 0.);
				vector<T> g_temp(4, 0.); //the value of the ugly expression div(p * grad(q_i)/rgqs), index is quaternion index
				T lmbda = 0.; //the lambda coefficient, comes from the lagrange multiplier.
				vector<T> deltaqi(4, 0.); //the rate equation w.r.t. time, for the various q_i components
				T M_q = 0.000001 + (M_qmax-0.000001)*(1-oldGrid(i)[0]*oldGrid(i)[0]*oldGrid(i)[0]*(10-15*oldGrid(i)[0]+6*oldGrid(i)[0]*oldGrid(i)[0]));
				
				for(int d = 0; d < 3; d++) {
					--x[d];
					phi_l[d] = oldGrid(x)[0];
					dq1_l[d] = dq(x)[0][d];
					dq2_l[d] = dq(x)[1][d];
					dq3_l[d] = dq(x)[2][d];
					dq4_l[d] = dq(x)[3][d];
					x[d] += 2;
					phi_r[d] = oldGrid(x)[0];
					dphi_cc[d] = 0.5*(phi_r[d]-phi_l[d])/deltax;
					--x[d];
					dqs_cc[d] = 0.25*((dq(x)[0][d]+dq1_l[d])*(dq(x)[0][d]+dq1_l[d]) + (dq(x)[1][d]+dq2_l[d])*(dq(x)[1][d]+dq2_l[d]) + (dq(x)[2][d]+dq3_l[d])*(dq(x)[2][d]+dq3_l[d]) + (dq(x)[3][d]+dq4_l[d])*(dq(x)[3][d]+dq4_l[d]));
				}
				lq[0] = ((dq(x)[0][0] - dq1_l[0]) + (dq(x)[0][1] - dq1_l[1]) + (dq(x)[0][2] - dq1_l[2]))/deltax;
				lq[1] = ((dq(x)[1][0] - dq2_l[0]) + (dq(x)[1][1] - dq2_l[1]) + (dq(x)[1][2] - dq2_l[2]))/deltax;
				lq[2] = ((dq(x)[2][0] - dq3_l[0]) + (dq(x)[2][1] - dq3_l[1]) + (dq(x)[2][2] - dq3_l[2]))/deltax;
				lq[3] = ((dq(x)[3][0] - dq4_l[0]) + (dq(x)[3][1] - dq4_l[1]) + (dq(x)[3][2] - dq4_l[2]))/deltax;
				rgqs_l[0] = max(sqrt(dq1_l[0]*dq1_l[0] + dq2_l[0]*dq2_l[0] + dq3_l[0]*dq3_l[0] + dq4_l[0]*dq4_l[0] + dqs_cc[1] + dqs_cc[2]), 1.);
				rgqs_l[1] = max(sqrt(dq1_l[1]*dq1_l[1] + dq2_l[1]*dq2_l[1] + dq3_l[1]*dq3_l[1] + dq4_l[1]*dq4_l[1] + dqs_cc[0] + dqs_cc[2]), 1.);
				rgqs_l[2] = max(sqrt(dq1_l[2]*dq1_l[2] + dq2_l[2]*dq2_l[2] + dq3_l[2]*dq3_l[2] + dq4_l[2]*dq4_l[2] + dqs_cc[0] + dqs_cc[1]), 1.);
				rgqs_r[0] = max(sqrt(dq(x)[0][0]*dq(x)[0][0] + dq(x)[1][0]*dq(x)[1][0] + dq(x)[2][0]*dq(x)[2][0] + dq(x)[3][0]*dq(x)[3][0] + dqs_cc[1] + dqs_cc[2]), 1.);
				rgqs_r[1] = max(sqrt(dq(x)[0][1]*dq(x)[0][1] + dq(x)[1][1]*dq(x)[1][1] + dq(x)[2][1]*dq(x)[2][1] + dq(x)[3][1]*dq(x)[3][1] + dqs_cc[0] + dqs_cc[2]), 1.);
				rgqs_r[2] = max(sqrt(dq(x)[0][2]*dq(x)[0][2] + dq(x)[1][2]*dq(x)[1][2] + dq(x)[2][2]*dq(x)[2][2] + dq(x)[3][2]*dq(x)[3][2] + dqs_cc[0] + dqs_cc[1]), 1.);
				for(int d = 0; d < 3; d++) {
					g_temp[0] += (dq(x)[0][d]*H*temp*(oldGrid(x)[0]*oldGrid(x)[0]+phi_r[d]*phi_r[d])/rgqs_r[d] - dq1_l[d]*H*temp*(oldGrid(x)[0]*oldGrid(x)[0]+phi_l[d]*phi_l[d])/rgqs_l[d]);
					g_temp[1] += (dq(x)[1][d]*H*temp*(oldGrid(x)[0]*oldGrid(x)[0]+phi_r[d]*phi_r[d])/rgqs_r[d] - dq2_l[d]*H*temp*(oldGrid(x)[0]*oldGrid(x)[0]+phi_l[d]*phi_l[d])/rgqs_l[d]);
					g_temp[2] += (dq(x)[2][d]*H*temp*(oldGrid(x)[0]*oldGrid(x)[0]+phi_r[d]*phi_r[d])/rgqs_r[d] - dq3_l[d]*H*temp*(oldGrid(x)[0]*oldGrid(x)[0]+phi_l[d]*phi_l[d])/rgqs_l[d]);
					g_temp[3] += (dq(x)[3][d]*H*temp*(oldGrid(x)[0]*oldGrid(x)[0]+phi_r[d]*phi_r[d])/rgqs_r[d] - dq4_l[d]*H*temp*(oldGrid(x)[0]*oldGrid(x)[0]+phi_l[d]*phi_l[d])/rgqs_l[d]);
				}
				T t1 = eqbar2*lq[0] + g_temp[0] + qf_comp(i)[0];
				T t2 = eqbar2*lq[1] + g_temp[1] + qf_comp(i)[1];
				T t3 = eqbar2*lq[2] + g_temp[2] + qf_comp(i)[2];
				T t4 = eqbar2*lq[3] + g_temp[3] + qf_comp(i)[3];
				lmbda = oldGrid(x)[2]*t1 + oldGrid(x)[3]*t2 + oldGrid(x)[4]*t3 + oldGrid(x)[5]*t4;
				deltaqi[0] = M_q*(t1 - oldGrid(x)[2]*lmbda);
				deltaqi[1] = M_q*(t2 - oldGrid(x)[3]*lmbda);
				deltaqi[2] = M_q*(t3 - oldGrid(x)[4]*lmbda);
				deltaqi[3] = M_q*(t4 - oldGrid(x)[5]*lmbda);
				newGrid(i)[2] = oldGrid(i)[2] + dt*deltaqi[0];
				newGrid(i)[3] = oldGrid(i)[3] + dt*deltaqi[1];
				newGrid(i)[4] = oldGrid(i)[4] + dt*deltaqi[2];
				newGrid(i)[5] = oldGrid(i)[5] + dt*deltaqi[3];
			}
			
			swap(oldGrid,newGrid);
		}
	}
	ghostswap(oldGrid);
	++iterations;
}
} //namespace MMSP