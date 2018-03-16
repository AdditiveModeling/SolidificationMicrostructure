namespace MMSP
{

template <int dim, typename T> void update(grid<dim,vector<T> >& oldGrid, int steps, MMSP::vector<double> T_vector, double step_size)
{
	std::cout<<step_size<<"\n";
	int id = 0;
	int np = 1;
	if (steps>T_vector.length()){
		std::cout<<"Requested number of steps not supported by T.txt";
		exit(1);
	}
	static int iterations = 1;
	#ifdef MPI_VERSION
	id = MPI::COMM_WORLD.Get_rank();
	np = MPI::COMM_WORLD.Get_size();
	#endif

	ghostswap(oldGrid);

	grid<dim,vector<T> > newGrid(oldGrid);
	static grid<dim,T>   refGrid(oldGrid, 0);  // create scalar- from vector-valued grid
	for (int i = 0; i < nodes(refGrid); i++) // initialize values for refGrid
		refGrid(i) = oldGrid(i)[0];

	const double     D_S = 1e-9;  	// Solid diffusion, cm^2/s
	const double     D_L = 1e-5;  	// Liquid diffusion, cm^2/s
	double     T_0 = T_vector[0];  	// Temperature (isothermal simulation)
	const double theta_0 = 0.;    	// angle relative to lab frame
	const double    T_mA = 1728.;	// Melting point of Nickel, K
	const double    T_mB = 1358.;	// Melting point of Copper, K
	const double     L_A = 2350.;	// Latent heat of Nickel, J/cm^3
	const double     L_B = 1728.;	// Latent heat of Copper, J/cm^3
	const double     s_A = 3.7e-5;	// Surface energy of Nickel, J/cm^2
	const double     s_B = 2.9e-5;	// Surface energy of Copper, J/cm^2
	const double     B_A = 0.33; 	// Linear kinetic coefficient of Nickel, cm/K/s
	const double     B_B = 0.39;	// Linear kinetic coefficient of Copper, cm/K/s
	const double     v_m = 7.42; 	// Molar volume, mol/cm^3
	const double       R = 8.314;   // Gas constant, J/(mol*K)
	const double     y_e = 0.06;    // anisotropy
	const double   alpha = 0.3;     // randomness at interface
	
	const double    d = dx(oldGrid,0)/0.94; // interfacial thickness
	const double  r72 = 6.*sqrt(2); 		//convenience var for 6sqrt(2)
	const double   dt = step_size;
	
	const double ebar = sqrt(r72*s_A*d/T_mA);
	const double  eb2 = ebar*ebar;
	const double  W_A = 3*s_A/(sqrt(2)*T_mA*d);
	const double  W_B = 3*s_B/(sqrt(2)*T_mB*d);
	const double  M_A = T_mA*T_mA*B_A/(r72*L_A*d);
	const double  M_B = T_mB*T_mB*B_B/(r72*L_B*d);
	
	std::cout.precision(2);
	
	
	std::cout<<dt<<"\n";
	std::cout << W_A << ", " << W_B << "\n";
	
	ghostswap(oldGrid);

	int minus = 0;
	int plus = 0;
	for (int step = 0; step < steps; step++) {
		if (id == 0)
			print_progress(step, steps);

		T_0=T_vector[step];
		//if (T_0>T_mA) T_0=T_mA-100;
		//if (T_0<T_mB) T_0=T_mB+100;

		grid<dim,vector<T> > gradphi(oldGrid);
		grid<dim,vector<T> > gradc(oldGrid);
		grid<dim,vector<T> > graddc(oldGrid);
		grid<dim,vector<T> > g(oldGrid); //holds g, gp
		grid<dim,vector<T> > dc(oldGrid); //holds D_C

		for (int i = 0; i < nodes(oldGrid); i++) {
			vector<int> x = position(oldGrid,i);
			for (int d = 0; d < dim; d++) {
				x[d]++;
				const T& right_p = oldGrid(x)[0];
				const T& right_c = oldGrid(x)[1];
				const T& right_d = D_S+(right_p*right_p*right_p*(6*right_p*right_p-15*right_p+10))*(D_L-D_S);
				x[d] -= 2;
				const T& left_p = oldGrid(x)[0];
				const T& left_c = oldGrid(x)[1];
				const T& left_d = D_S+(left_p*left_p*left_p*(6*left_p*left_p-15*left_p+10))*(D_L-D_S);
				x[d]++;
				gradphi(i)[d] = (right_p - left_p) / (2*dx(oldGrid,d));
				gradc(i)[d] = (right_c - left_c) / (2*dx(oldGrid,d));
				graddc(i)[d] = (right_d - left_d) / (2*dx(oldGrid,d));				
			}
			g(i)[0] = oldGrid(x)[0]*oldGrid(x)[0]*(1-oldGrid(x)[0])*(1-oldGrid(x)[0]);
			g(i)[1] = 4*oldGrid(x)[0]*oldGrid(x)[0]*oldGrid(x)[0] - 6*oldGrid(x)[0]*oldGrid(x)[0] + 2*oldGrid(x)[0];
			dc(i)[0] = D_S+(oldGrid(x)[0]*oldGrid(x)[0]*oldGrid(x)[0]*(6*oldGrid(x)[0]*oldGrid(x)[0]-15*oldGrid(x)[0]+10))*(D_L-D_S);
		}

		// Sync parallel grids
		ghostswap(gradphi);
		ghostswap(gradc); //is this needed?
		ghostswap(graddc); //is this needed?
		ghostswap(g);
		ghostswap(dc);

		for (int i = 0; i < nodes(oldGrid); i++) {
			vector<int> x = position(oldGrid,i);
			const vector<T>& oldX = oldGrid(x);
			const vector<T>&  gpX = gradphi(x);
			      vector<T>& newX = newGrid(x);
				  vector<T> d2phi(dim,0.);
				  vector<T> d2c(dim,0.);
				  vector<T> gradtemp(dim,0.);
				  
			for (int d = 0; d < dim; d++) {
				x[d]++;
				const T& right_p = oldGrid(x)[0];
				const T& right_c = oldGrid(x)[1];
				const T& right_temp = dc(x)[0]*v_m*oldGrid(x)[1]*(1-oldGrid(x)[1])*((W_B-W_A)*g(x)[1] + 30*g(x)[0]*(L_B*(1/T_0-1/T_mB)-L_A*(1/T_0-1/T_mA))/R);
				x[d] -= 2;
				const T& left_p = oldGrid(x)[0];
				const T& left_c = oldGrid(x)[1];
				const T& left_temp = dc(x)[0]*v_m*oldGrid(x)[1]*(1-oldGrid(x)[1])*((W_B-W_A)*g(x)[1] + 30*g(x)[0]*(L_B*(1/T_0-1/T_mB)-L_A*(1/T_0-1/T_mA))/R);
				x[d]++;
				d2phi[d] = (right_p + left_p - 2*oldX[0]) / (dx(oldGrid,d)*dx(oldGrid,d));
				d2c[d] = (right_c + left_c - 2*oldX[1]) / (dx(oldGrid,d)*dx(oldGrid,d));
				gradtemp[d] = (right_temp - left_temp) / (2*dx(oldGrid,d));
			}
			const T lapPhi = d2phi[0]+d2phi[1];
			const T lapC = d2c[0]+d2c[1];
			      T phixy;
			x[0]++;
			const T& rightgradphi = gradphi(x)[1];
			x[0] -= 2;
			const T& leftgradphi = gradphi(x)[1];
			x[0]++;
			phixy = (rightgradphi - leftgradphi) / (2*dx(oldGrid,d));
			const double theta = theta_0+atan2(gradphi(x)[1], gradphi(x)[0]);
			const double eta   = 1+y_e*cos(4*theta);
			const double etap  = -4*y_e*sin(4*theta);
			const double etapp = -16*(eta-1);
			const double c2    = cos(2*theta);
			const double s2    = sin(2*theta);
			const double M_phi = (1-oldX[1])*M_A + oldX[1]*M_B;
			const double r     = 2*((double) rand() / (RAND_MAX))-1; //random # between -1 and 1
			
			const double H_A   = W_A*g(x)[1] + 30*L_A*(1/T_0-1/T_mA)*g(x)[0];
			const double H_B   = W_B*g(x)[1] + 30*L_B*(1/T_0-1/T_mB)*g(x)[0];
			const double temp = dc(x)[0]*v_m*oldX[1]*(1-oldX[1])*(H_B-H_A)/R;
			
			const double deltac = dc(x)[0]*lapC + graddc(x)[0]*gradc(x)[0] + graddc(x)[1]*gradc(x)[1] + temp*lapPhi + gradphi(x)[0]*gradtemp[0] + gradphi(x)[1]*gradtemp[1];
				  double deltaphi = M_phi*((eb2*eta*eta*lapPhi - (1-oldX[1])*H_A - oldX[1]*H_B) + (eb2*eta*etap*(s2*(d2phi[1]-d2phi[0]) + 2*c2*phixy)) + (0.5*eb2*(etap*etap+eta*etapp)*(-2*s2*phixy+lapPhi+c2*(d2phi[1]-d2phi[0]))));
			const double interfacial_randomness = M_phi*alpha*r*(16*g(x)[0])*((1-oldX[1])*H_A + oldX[1]*H_B);
			deltaphi += interfacial_randomness;
			newX[0] = oldX[0] + deltaphi*dt;
			newX[1] = oldX[1] + deltac*dt;
			refGrid(i) = oldX[0];
			/*
			if(!(oldX[0] == 0. || oldX[0] == 1.)) {
				std::cout << deltac << ", ";
				std::cout << deltaphi << ", ";
				std::cout << lapPhi << ", ";
				std::cout << lapC << ", ";
				std::cout << temp << ", ";
				std::cout << H_A << ", ";
				std::cout << W_A*g(x)[1] << ", ";
				std::cout << g(x)[0] << ", ";
				std::cout << graddc(x)[0] << ", ";
				std::cout << graddc(x)[1] << ", ";
				std::cout << gradtemp[0] << ", ";
				std::cout << gradtemp[1] << ", ";
				std::cout << gradphi(x)[0] << ", ";
				std::cout << gradphi(x)[1] << "\n";
				
			}
			*/
		}

		swap(oldGrid,newGrid);
		ghostswap(oldGrid);
	}

	iterations++;
}

} // namespace MMSP