#ifndef __NUMERICAL_GRID_H__
#define __NUMERICAL_GRID_H__

#include <assert.h>
#include "Util.hpp"

namespace CQP{

template<unsigned N>
class GridPoint {
public:

	GridPoint() {
		for (int i = 0; i < N; i++) {
			points[i] = 0;
		}
	}

	template<typename... Args>
	GridPoint(Args... parameters)
		: parameter_count(sizeof...(parameters)) {
		assert(sizeof...(parameters) == N);
		init(parameters...);
	}

	GridPoint(const GridPoint<N>& p) {
		for (int i = 0; i < N; i++) {
			points[i] = p.points[i];
		}
	}

	GridPoint<N>& operator=(const GridPoint<N>& p) {
		for (int i = 0; i < N; i++) {
			points[i] = p.points[i];
		}
		return *this;
	}

	friend bool operator==(const GridPoint<N>& a, const GridPoint<N>& b) {
		bool flag = true;
		for (int i = 0; i < N; i++) {
			flag = (a.points[i] == b.points[i]) && flag;
		}
		return flag;
	}

	double& get(int degree) {
		assert(degree < N);
		return points[degree];
	}

	double& x() {
		assert(N >= 1);
		return points[0];
	}

	double& y() {
		assert(N >= 2);
		return points[1];
	}

	double& z() {
		assert(N >= 3);
		return points[2];
	}

	friend std::ostream& operator<<(std::ostream& os, const GridPoint<N>& p) {
		os << "(";
		for (int i = 0; i < N; i++) {
			os << p.points[i];
			if (i != N - 1) {
				os << ", ";
			}
		}
		os << ")";
		return os;
	}

private:
	void init(double num) {
		points[N - 1] = num;
	}

	template<typename... Args>
	void init(double num, Args... rest) {
		int init_update_pos = parameter_count - (sizeof...(rest) + 1);
		points[init_update_pos] = num;
		init(rest...);
	}

	int parameter_count;
	double points[N];
};

typedef GridPoint<1> Point1D;
typedef GridPoint<2> Point2D;
typedef GridPoint<3> Point3D;

template<unsigned N>
struct GridIndice
{
	constexpr static int size = N;
	constexpr static int N_minus_one = N - 1;
	int indice[N];

	GridIndice()
		{ for(int i = 0; i < N; i++) indice[i] = 0; }

	int& operator[](int degree){
		return indice[degree];
	}
	
	GridIndice<N_minus_one> erase(int degree){
		GridIndice<N_minus_one> sub_grid;
		int k = 0;
		for(int i = 0; i < N; i++){
			if(i != degree){
				sub_grid[k] = indice[i];
				k = k + 1;
			}
		}
		return sub_grid;
	}

	template<unsigned _N>
	GridIndice<_N> sub(int start_id, int end_id)
	{
		assert(end_id - start_id == _N);
		GridIndice<_N> sub_grid;
		int k = 0;
		for(int i = start_id; i < end_id; i++){
			sub_grid[k] = indice[i];
			k = k + 1;
		}
		return sub_grid;
	}
};

constexpr int LITTEL_INDEX_FASTRUNING = 1;
constexpr int LARGE_INDEX_FASTRUNING = 2;

constexpr int X_DEGREE = 0;
constexpr int Y_DEGREE = 1;
constexpr int Z_DEGREE = 2;

constexpr int R_DEGREE = 0;
constexpr int THETA_DEGREE = 1;
constexpr int PHI_DEGREE = 2;

//DO NOT USE M = LARGE_INDEX_FASTRUNING!!!!!!!!

template<unsigned N, unsigned M = LITTEL_INDEX_FASTRUNING>
class NumericalGrid
{
public:
	constexpr static int N_plus_one = N + 1;

	template<typename... Args>
	NumericalGrid(Args... parameters)
		: parameter_count(static_cast<int>(sizeof...(parameters) / 3))
	{
		assert(sizeof...(parameters) == (3 * N));
		init(parameters...);
		update_total_cnt();
	}

	NumericalGrid(int(&cnt)[N], double(&len)[N], double(&off)[N]) : parameter_count(N)
	{
		for (int i = 0; i < N; i++) {
			pos_count[i] = cnt[i];
			length[i] = len[i];
			offset[i] = off[i];
		}
		update_total_cnt();
	}

	NumericalGrid() : parameter_count(N) {
		for (int i = 0; i < N; i++) {
			pos_count[i] = 0;
			length[i] = 0.0;
			offset[i] = 0.0;
		}
		update_total_cnt();
	}

	NumericalGrid(const NumericalGrid<N>& grid) {
		for (int i = 0; i < N; i++) {
			pos_count[i] = grid.pos_count[i];
			length[i] = grid.length[i];
			offset[i] = grid.offset[i];
		}
		update_total_cnt();
	}

	NumericalGrid<N>& operator=(const NumericalGrid<N>& grid) {
		for (int i = 0; i < N; i++) {
			pos_count[i] = grid.pos_count[i];
			length[i] = grid.length[i];
			offset[i] = grid.offset[i];
		}
		update_total_cnt();
		return *this;
	}

	friend bool operator==(const NumericalGrid<N>& a, const NumericalGrid<N>& b) {
		bool flag = true;
		for (int i = 0; i < N; i++) {
			flag = (isApproxEqual(a.pos_count[i], b.pos_count[i])
				 && isApproxEqual(a.length[i], b.length[i])
				 && isApproxEqual(a.offset[i], b.offset[i])) && flag;
		}
		return flag;
	}

	GridPoint<N> index(int id) const { return index_agent(id); }

	GridIndice<N> expand(int id) const 
	{
		assert(id >= 0 && id < total_count);
		GridIndice<N> ans;
		for (int j = 0; j < N; j++) {
			int num = id % pos_count[j];
			ans[j] = num;
			id = (id - num) / pos_count[j];
		}
		return ans;
	}

	int shrink(GridIndice<N>& indices) const {
		assert(indices.size == N);
		int id = 0, tmp = 1;
		for (int j = 0; j < N; j++) {
			if (indices[j] < 0 || indices[j] >= pos_count[j]) {
				return -1;
			}
			id += indices[j] * tmp;
			tmp *= pos_count[j];
		}
		return id;
	}

	int shrink(const std::vector<int>& indices) const {
		assert(indices.size() == N);
		int id = 0, tmp = 1;
		for (int j = 0; j < N; j++) {
			if (indices[j] < 0 || indices[j] >= pos_count[j]) {
				return -1;
			}
			id += indices[j] * tmp;
			tmp *= pos_count[j];
		}
		return id;
	}

	int shrink(int (&indices)[N]) const {
		int id = 0, tmp = 1;
		for (int j = 0; j < N; j++) {
			if (indices[j] < 0 || indices[j] >= pos_count[j]) {
				return -1;
			}
			id += indices[j] * tmp;
			tmp *= pos_count[j];
		}
		return id;
	}

	template<unsigned _N>
	NumericalGrid<_N> subset(std::initializer_list<int> bitset) const
	{
		assert(bitset.size() == _N && _N <= N && _N > 0);
		int new_pos_cnt[_N] = { 0 };
		double new_len[_N] = { 0.0 };
		double new_off[_N] = { 0.0 };

		int k = 0;
		for (auto iter = bitset.begin(); iter != bitset.end(); iter++) {
			new_pos_cnt[k] = pos_count[*iter];
			new_len[k] = length[*iter];
			new_off[k] = offset[*iter];
			k++;
		}

		NumericalGrid<_N> new_grid(new_pos_cnt, new_len, new_off);
		return new_grid;
	}

	template<unsigned _N>
	NumericalGrid<_N> subset(std::vector<int> bitset) const
	{
		assert(bitset.size() == _N && _N <= N && _N > 0);
		int new_pos_cnt[_N] = { 0 };
		double new_len[_N] = { 0.0 };
		double new_off[_N] = { 0.0 };

		int k = 0;
		for (auto iter = bitset.begin(); iter != bitset.end(); iter++) {
			new_pos_cnt[k] = pos_count[*iter];
			new_len[k] = length[*iter];
			new_off[k] = offset[*iter];
			k++;
		}

		NumericalGrid<_N> new_grid(new_pos_cnt, new_len, new_off);
		return new_grid;
	}

	NumericalGrid<N_plus_one> supersetOnce(int id, int cnt, double len, double off) const {
		int new_pos_cnt[N_plus_one] = { 0 };
		double new_len[N_plus_one] = { 0.0 };
		double new_off[N_plus_one] = { 0.0 };
		int k = 0;
		for (int i = 0; i < N_plus_one; i++) {
			if (i == id) {
				new_pos_cnt[i] = cnt;
				new_len[i] = len;
				new_off[i] = off;
			}
			else {
				new_pos_cnt[i] = pos_count[k];
				new_len[i] = length[k];
				new_off[i] = offset[k];
				k++;
			}
		}
		return NumericalGrid<N_plus_one>(new_pos_cnt, new_len, new_off);
	}

	NumericalGrid<N> scramble(std::initializer_list<int> bitset)
	{
		//check validity of input
		std::vector<int> normal_order(N);
		for (int i = 0; i < N; i++) {
			normal_order[i] = i;
		}
		if (!is_permutation(normal_order.begin(), normal_order.end(), bitset.begin())) {
			return NumericalGrid<N>();
		}

		int new_pos_cnt[N] = { 0 };
		double new_len[N] = { 0.0 };
		double new_off[N] = { 0.0 };
		int k = 0;
		for (auto degree : bitset) {
			new_pos_cnt[degree] = k;
			new_len[degree] = k;
			new_off[degree] = k;
			k++;
		}
		return NumericalGrid<N>(new_pos_cnt, new_len, new_off);
	}

	int getTotalCount() const { return total_count; }
	int getCount(int degree = 0) const { return pos_count[degree]; }
	double getLength(int degree = 0) const { return length[degree]; }
	double getOffset(int degree = 0) const { return offset[degree]; }
	double getDelta(int degree = 0) const { return length[degree] / pos_count[degree]; }

	friend std::ostream& operator<<(std::ostream& os, const NumericalGrid<N>& grid) {
		os << "pos_count: ";
		for (int i = 0; i < N; i++) {
			os << grid.pos_count[i] << " ";
		}
		os << std::endl;

		os << "length: ";
		for (int i = 0; i < N; i++) {
			os << grid.length[i] << " ";
		}
		os << std::endl;

		os << "offset: ";
		for (int i = 0; i < N; i++) {
			os << grid.offset[i] << " ";
		}
		os << std::endl;
		return os;
	}

private:
	double get_value_by_index(int index, int degree) const {
		return offset[degree] - length[degree] / 2 + index * (length[degree] / pos_count[degree]);
	}

	void update_total_cnt() {
		total_count = 1;
		for (int i = 0; i < N; i++) {
			total_count *= pos_count[i];
		}
	}

	void init(int num, double len, double off) {
		pos_count[N - 1] = num;
		length[N - 1] = len;
		offset[N - 1] = off;
	}

	template<typename... Args>
	void init(int num, double len, double off, Args... rest) {
		int init_update_pos = parameter_count - (static_cast<int>(sizeof...(rest) / 3) + 1);
		pos_count[init_update_pos] = num;
		length[init_update_pos] = len;
		offset[init_update_pos] = off;
		init(rest...);
	}

	GridPoint<N> index_agent(int id) const {
		assert(id >= 0 && id < total_count);
		GridPoint<N> ans;
		for (int j = 0; j < N; j++) {
			int num = id % pos_count[j];
			ans.get(j) = get_value_by_index(num, j);
			id = (id - num) / pos_count[j];
		}
		return ans;
	}

private:
	int pos_count[N];
	double length[N];
	double offset[N];
	int parameter_count;
	int total_count;
};

typedef NumericalGrid<1> NumericalGrid1D;
typedef NumericalGrid<2> NumericalGrid2D;
typedef NumericalGrid<3> NumericalGrid3D;

}


#endif // !__NUMERICAL_GRID_H__