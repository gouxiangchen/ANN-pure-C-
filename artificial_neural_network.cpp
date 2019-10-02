#include <iostream>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <vector>

using namespace std;

class Util
{
public:
    static void random_seed(int seed = 0)
    {
        srand(seed);
    }
    static double random()
    {
        return ((rand() % 10000) - 5000) / 5000.;
    }
};

class Matrix
{
public:
    Matrix(const Matrix & m);
    Matrix(int r, int c, bool random=true);  // initial a random matrix
    Matrix multiple(const Matrix & another_matrix) const;
    Matrix(const int * m, int r, int c);
    Matrix(){rows = 0; cols = 0; matrix = nullptr;}
    ~Matrix();
    void print();
    friend Matrix operator + (double scallar, const Matrix & m);
    Matrix operator / (double scallar) const;
    friend Matrix operator * (double scallar, const Matrix & m);
    void plus_(const Matrix & another_matrix);  // _ stands for in-place operation, change the current object
    void zeros_();
    void divide_scallar_(double scallar);
    void multiple_scallar_(double scallar);
    void bias_(const Matrix & bias_matrix);  // bias matrix must be a vector, the first dimension should be 1
    Matrix & operator = (const Matrix & another_matrix);
    int get_batch() const {return rows;}
    int get_features() const {return cols;}
    int * get_matrix() const {return matrix;}
    Matrix t() const;
    void show_shape() const;
private:
    int rows;
    int cols;
    double * matrix;
};

Matrix operator+(double scallar, const Matrix & m)
{
	Matrix out(m.rows, m.cols, false);
	for (int y = 0; y < m.rows; y ++){
        for (int x = 0; x < m.cols; x++){
            out.matrix[y * m.cols + x] = m.matrix[y * m.cols + x] + scallar;
        }
    }
    return out;
}

Matrix operator*(double scallar, const Matrix & m)
{
	Matrix out(m.rows, m.cols, false);
	for (int y = 0; y < m.rows; y ++){
        for (int x = 0; x < m.cols; x++){
            out.matrix[y * m.cols + x] = m.matrix[y * m.cols + x] * scallar;
        }
    }
    return out;
}

Matrix Matrix::operator/(double scallar) const
{
	Matrix out(this->rows, this->cols, false);
	for (int y = 0; y < rows; y ++){
        for (int x = 0; x < cols; x++){
            out.matrix[y * cols + x] = this->matrix[y * cols + x] / scallar;
        }
    }
    return out;
}

Matrix& Matrix::operator=(const Matrix & another_matrix)
{
    if (this == &another_matrix){
        return *this;
    }
    cout << "delete : " << rows << " * " << cols << endl;
    rows = another_matrix.rows;
    cols = another_matrix.cols;
    delete [] matrix;
    matrix = new double[rows * cols];
    cout << "new " << rows << " * " << cols << " memory" << endl;
    for (int y = 0; y < rows; y ++){
        for (int x = 0; x < cols; x++){
            this->matrix[y * cols + x] = another_matrix.matrix[y * cols + x];
        }
    }
    return *this;
}

Matrix::Matrix(const int * m, int r, int c)
{
    this->rows = r;
    this->cols = c;
    matrix = new double[rows * cols];
    cout << "new " << rows << " * " << cols << " memory" << endl;
    for (int y = 0; y < rows; y ++){
        for (int x = 0; x < cols; x++){
            this->matrix[y * cols + x] = m[y * cols + x];
        }
    }
}

Matrix::Matrix(const Matrix & m)
{
    this->rows = m.rows;
    this->cols = m.cols;
    matrix = new double[rows * cols];
    cout << "new " << rows << " * " << cols << " memory" << endl;
    for (int y = 0; y < rows; y ++){
        for (int x = 0; x < cols; x++){
            matrix[y * cols + x] = m.matrix[y * cols + x];
        }
    }
}

Matrix::Matrix(int r, int c, bool random)
{
    this->rows = r;
    this->cols = c;
	this->matrix = new double[rows * cols];
    cout << "new " << rows << " * " << cols << " memory" << endl;
    if (random){
        for (int y = 0; y < rows; y ++){
            for (int x = 0; x < cols; x++){
                matrix[y * cols + x] = Util::random();
            }
        }
    }
    else {
        memset(matrix, 0, sizeof(double) * rows * cols);
    }

}

Matrix Matrix::multiple(const Matrix & another_matrix) const
{
    assert(this->cols == another_matrix.rows);
    Matrix result(this->rows, another_matrix.cols, false);
    for (int y = 0; y < this->rows; y ++){
        for (int x = 0; x < another_matrix.cols; x++){
            double res = 0;
            for (int temp = 0; temp < this->cols; temp ++){
                res += (this->matrix[y * this->cols + temp] * another_matrix.matrix[temp * another_matrix.cols + x]);
            }
            result.matrix[y * another_matrix.cols + x] = res;
        }
    }
    return result;
}

void Matrix::plus_(const Matrix & another_matrix)
{
    assert((this->rows == another_matrix.rows) && (this->cols == another_matrix.cols));
    for (int y = 0; y < rows; y ++){
        for (int x = 0; x < cols; x++){
            this->matrix[y * cols + x] += another_matrix.matrix[y * cols + x];
        }
    }
}

void Matrix::zeros_()
{
    memset(matrix, 0, sizeof(double) * rows * cols);
}

void Matrix::print()
{
    for (int y = 0; y < rows; y ++){
        for (int x = 0; x < cols; x++){
            cout << this->matrix[y * cols + x] << "\t";
        }
        cout << endl;
    }
}

void Matrix::multiple_scallar_(double scallar)
{
    for (int y = 0; y < rows; y ++){
        for (int x = 0; x < cols; x++){
            this->matrix[y * cols + x] *= scallar;
        }
    }
}

void Matrix::divide_scallar_(double scallar)
{
    for (int y = 0; y < rows; y ++){
        for (int x = 0; x < cols; x++){
            this->matrix[y * cols + x] /= scallar;
        }
    }
}

void Matrix::bias_(const Matrix & bias_matrix)
{
    assert(bias_matrix.rows == 1 && bias_matrix.cols == this->cols);
    for (int y = 0; y < rows; y ++){
        for (int x = 0; x < cols; x++){
            this->matrix[y * cols + x] += bias_matrix.matrix[x];
        }
    }
}

Matrix Matrix::t() const
{
    Matrix res(cols, rows, false);
    for (int y = 0; y < rows; y ++){
        for (int x = 0; x < cols; x++){
            res.matrix[x * rows + y] = this->matrix[y * cols + x];
        }
    }
    return res;
}

void Matrix::show_shape() const
{
    cout << "(" << rows << ", " << cols << ")" << endl;
}

Matrix::~Matrix()
{
    cout << "delete : " << rows << " * " << cols << endl;
//    print();
    delete [] matrix;
}

class Layer_base
{
public:
    virtual Matrix forward(const Matrix & input)=0;
    virtual Matrix backward(const Matrix & pre_grad)=0;
    virtual void update_weight(double learn_rate=0.001)=0;
};

class Linear_layer: public Layer_base
{
public:
    Linear_layer(int input_features, int output_features);
    virtual Matrix forward(const Matrix & input);
    virtual Matrix backward(const Matrix & pre_grad);
    virtual void update_weight(double learn_rate=0.001);
    ~Linear_layer();
private:
    void zero_grad_();
    Linear_layer(const Linear_layer &);
    Matrix weight;
    Matrix grad;
    Matrix bias;
    Matrix bias_grad;
    Matrix input;
    Matrix output;
};

Linear_layer::Linear_layer(int input_features, int output_features)
    :weight(input_features, output_features), grad(input_features, output_features, false),
    bias(1, output_features), bias_grad(1, output_features)
{

}

Linear_layer::~Linear_layer()
{
//    cout << "delete Linear" << endl;
}

void Linear_layer::zero_grad_()
{
    grad.zeros_();
}

void Linear_layer::update_weight(double learn_rate)
{
    grad.multiple_scallar_(learn_rate);
    bias_grad.multiple_scallar_(learn_rate);
    weight.plus_(grad);
    bias.plus_(bias_grad);
    zero_grad_();
}

Matrix Linear_layer::forward(const Matrix & input)
{
    this->input = input;
    Matrix output = input.multiple(weight);
    output.bias_(bias);
    return output;
}

Matrix Linear_layer::backward(const Matrix & pre_grad)
{
    int batch_size = this->input.get_batch();
    this->grad = this->input.t().multiple(pre_grad);
    this->grad.divide_scallar_(batch_size);
    this->bias_grad = pre_grad;
    Matrix next_grad = pre_grad.multiple(this->weight.t());
    return next_grad;
}

class ReLU_layer : public Layer_base
{
public:
	virtual Matrix forward(const Matrix & input);
	virtual Matrix backward(const Matrix & pre_grad);
	virtual void update_weight(double learn_rate = 0.001){}	// relu layer has no weight
	ReLU_layer(){}
	~ReLU_layer(){}
private:
	Matrix input;
};

Matrix ReLU_layer::forward(const Matrix& input)
{
	this->input = input;
	Matrix output = input;
	int rows = input.get_batch();
	int cols = input.get_features();
	int * input_matrix = input.get_matrix();
	int * output_matrix = output.get_matrix();
	for (int y = 0; y < rows; y ++){
        for (int x = 0; x < cols; x++){
			if (input_matrix[y * cols + x] > 0){
				output_matrix[y * cols + x] = input_matrix[y * cols + x];
			}
            else {
				output_matrix[y * cols + x] = 0;
            }
        }
    }
    return output;
}

Matrix ReLU_layer::backward(const Matrix& pre_grad)
{
	assert((pre_grad.get_batch() == input.get_batch()) && (pre_grad.get_features() == input.get_features()));
	Matrix next_grad = pre_grad;
	int * input_matrix = this->input.get_matrix();
	int * next_grad_matrix = next_grad.get_matrix();
	for (int y = 0; y < rows; y ++){
        for (int x = 0; x < cols; x++){
			if (input_matrix[y * cols + x] > 0){
				next_grad_matrix[y * cols + x] = pre_grad[y * cols + x];
			}
            else {
				next_grad_matrix[y * cols + x] = 0;
            }
        }
    }
    return next_grad;
}

void no_more_memory()
{
	cout << "no more memory" << endl;
	exit(-1);
}

int main()
{
	set_new_handler(no_more_memory);
    int batch_size = 2;
    Util::random_seed(int(time(0)));
    vector<Layer_base *> network;
    Linear_layer fc1(5, 3);
    network.push_back(&fc1);
    Matrix input(batch_size, 5);
    Matrix output = input;
    input.print();

    for (int i = 0; i < network.size(); i++){
        output = network[i]->forward(output);
    }
    output.print();
    int loss_array[6] = {1, 1, 1, 1, 1, 1};
    Matrix loss_matrix(loss_array, 2, 3);
    Matrix pre_grad = loss_matrix/batch_size;
	for (int i = 0; i < network.size(); i++){
        pre_grad = network[i]->backward(pre_grad);
    }
    return 0;
}



