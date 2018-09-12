namespace MMSP {
	
	template <int dim, typename T> grid<dim,T> average_r(grid<dim,vector<T> >& oldGrid, int index, int d) {
		grid<dim,T> averageGrid(oldGrid, 0);
		for (int i=0; i<nodes(oldGrid); ++i) {
			vector<int> x=position(oldGrid,i);
			x[d]++;
			averageGrid(i) = (oldGrid(i)[index]+oldGrid(x)[index])*0.5;
		}
		return averageGrid;
	}
	template <int dim, typename T> grid<dim,T> average_r(grid<dim,T>& oldGrid, int d) {
		grid<dim,T> averageGrid(oldGrid);
		for (int i=0; i<nodes(oldGrid); ++i) {
			vector<int> x=position(oldGrid,i);
			x[d]++;
			averageGrid(i) = (oldGrid(i)+oldGrid(x))*0.5;
		}
		return averageGrid;
	}
	template <int dim, typename T> grid<dim,T> average_l(grid<dim,vector<T> >& oldGrid, int index, int d) {
		grid<dim,T> averageGrid(oldGrid, 0);
		for (int i=0; i<nodes(oldGrid); ++i) {
			vector<int> x=position(oldGrid,i);
			x[d]--;
			averageGrid(i) = (oldGrid(i)[index]+oldGrid(x)[index])*0.5;
		}
		return averageGrid;
	}
	template <int dim, typename T> grid<dim,T> average_l(grid<dim,T>& oldGrid, int d) {
		grid<dim,T> averageGrid(oldGrid);
		for (int i=0; i<nodes(oldGrid); ++i) {
			vector<int> x=position(oldGrid,i);
			x[d]--;
			averageGrid(i) = (oldGrid(i)+oldGrid(x))*0.5;
		}
		return averageGrid;
	}
	template <int dim, typename T> grid<dim,T> partial_r(grid<dim,vector<T>>& oldGrid, int index, int d, T deltax) {
		grid<dim,T> averageGrid(oldGrid, 0);
		T inverse_dx = 1./deltax;
		for (int i=0; i<nodes(oldGrid); ++i) {
			vector<int> x=position(oldGrid,i);
			x[d]++;
			averageGrid(i) = (oldGrid(x)[index]-oldGrid(i)[index])*inverse_dx;
		}
		return averageGrid;
	}
	template <int dim, typename T> grid<dim,T> partial_r(grid<dim,T>& oldGrid, int d, T deltax) {
		grid<dim,T> averageGrid(oldGrid);
		T inverse_dx = 1./deltax;
		for (int i=0; i<nodes(oldGrid); ++i) {
			vector<int> x=position(oldGrid,i);
			x[d]++;
			averageGrid(i) = (oldGrid(x)-oldGrid(i))*inverse_dx;
		}
		return averageGrid;
	}
	template <int dim, typename T> grid<dim,T> partial_l(grid<dim,vector<T>>& oldGrid, int index, int d, T deltax) {
		grid<dim,T> averageGrid(oldGrid, 0);
		T inverse_dx = 1./deltax;
		for (int i=0; i<nodes(oldGrid); ++i) {
			vector<int> x=position(oldGrid,i);
			x[d]--;
			averageGrid(i) = (oldGrid(i)[index]-oldGrid(x)[index])*inverse_dx;
		}
		return averageGrid;
	}
	template <int dim, typename T> grid<dim,T> partial_l(grid<dim,T>& oldGrid, int d, T deltax) {
		grid<dim,T> averageGrid(oldGrid);
		T inverse_dx = 1./deltax;
		for (int i=0; i<nodes(oldGrid); ++i) {
			vector<int> x=position(oldGrid,i);
			x[d]--;
			averageGrid(i) = (oldGrid(i)-oldGrid(x))*inverse_dx;
		}
		return averageGrid;
	}
}