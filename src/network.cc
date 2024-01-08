#include "./network.h"

void Network::forward(const Matrix &input)
{
  std::chrono::_V2::system_clock::time_point start, end;
  std::chrono::nanoseconds elapsed;
  if (layers.empty())
    return;

  if (layers[0]->is_custom_convolution())
    start = std::chrono::high_resolution_clock::now();

  layers[0]->forward(input);

  if (layers[0]->is_custom_convolution())
  {
    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "Forward Convolution - Layer 0 - Forward Time: " << elapsed.count() / 1000000000.0 << "s" << std::endl;
  }

  for (int i = 1; i < layers.size(); i++)
  {
    if (layers[i]->is_custom_convolution())
      start = std::chrono::high_resolution_clock::now();

    layers[i]->forward(layers[i - 1]->output());

    if (layers[i]->is_custom_convolution())
    {
      end = std::chrono::high_resolution_clock::now();
      elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
      std::cout << "Forward Convolution - Layer " << i << " - Forward Time: " << elapsed.count() / 1000000000.0 << "s" << std::endl;
    }
  }
}

void Network::backward(const Matrix &input, const Matrix &target)
{
  int n_layer = layers.size();
  // 0 layer
  if (n_layer <= 0)
    return;
  // 1 layer
  loss->evaluate(layers[n_layer - 1]->output(), target);
  if (n_layer == 1)
  {
    layers[0]->backward(input, loss->back_gradient());
    return;
  }
  // >1 layers
  layers[n_layer - 1]->backward(layers[n_layer - 2]->output(),
                                loss->back_gradient());
  for (int i = n_layer - 2; i > 0; i--)
  {
    layers[i]->backward(layers[i - 1]->output(), layers[i + 1]->back_gradient());
  }
  layers[0]->backward(input, layers[1]->back_gradient());
}

void Network::update(Optimizer &opt)
{
  for (int i = 0; i < layers.size(); i++)
  {
    layers[i]->update(opt);
  }
}

std::vector<std::vector<float>> Network::get_parameters() const
{
  const int n_layer = layers.size();
  std::vector<std::vector<float>> res;
  res.reserve(n_layer);
  for (int i = 0; i < n_layer; i++)
  {
    res.push_back(layers[i]->get_parameters());
  }
  return res;
}

void Network::set_parameters(const std::vector<std::vector<float>> &param)
{
  const int n_layer = layers.size();
  if (static_cast<int>(param.size()) != n_layer)
    throw std::invalid_argument("Parameter size does not match");
  for (int i = 0; i < n_layer; i++)
  {
    layers[i]->set_parameters(param[i]);
  }
}

std::vector<std::vector<float>> Network::get_derivatives() const
{
  const int n_layer = layers.size();
  std::vector<std::vector<float>> res;
  res.reserve(n_layer);
  for (int i = 0; i < n_layer; i++)
  {
    res.push_back(layers[i]->get_derivatives());
  }
  return res;
}

void Network::check_gradient(const Matrix &input, const Matrix &target,
                             int n_points, int seed)
{
  if (seed > 0)
    std::srand(seed);

  this->forward(input);
  this->backward(input, target);
  std::vector<std::vector<float>> param = this->get_parameters();
  std::vector<std::vector<float>> deriv = this->get_derivatives();

  const float eps = 1e-4;
  const int n_layer = deriv.size();
  for (int i = 0; i < n_points; i++)
  {
    // Randomly select a layer
    const int layer_id = int(std::rand() / double(RAND_MAX) * n_layer);
    // Randomly pick a parameter, note that some layers may have no parameters
    const int n_param = deriv[layer_id].size();
    if (n_param < 1)
      continue;
    const int param_id = int(std::rand() / double(RAND_MAX) * n_param);
    // Turbulate the parameter a little bit
    const float old = param[layer_id][param_id];

    param[layer_id][param_id] -= eps;
    this->set_parameters(param);
    this->forward(input);
    this->backward(input, target);
    const float loss_pre = loss->output();

    param[layer_id][param_id] += eps * 2;
    this->set_parameters(param);
    this->forward(input);
    this->backward(input, target);
    const float loss_post = loss->output();

    const float deriv_est = (loss_post - loss_pre) / eps / 2;

    std::cout << "[layer " << layer_id << ", param " << param_id << "] deriv = " << deriv[layer_id][param_id] << ", est = " << deriv_est << ", diff = " << deriv_est - deriv[layer_id][param_id] << std::endl;

    param[layer_id][param_id] = old;
  }

  // Restore original parameters
  this->set_parameters(param);
}

void Network::save_parameters(const std::string &filename)
{
  std::ofstream file(filename, std::ios::binary | std::ios::out | std::ios::trunc);
  uint64_t expected_file_size = 0;

  if (file.is_open() == false)
  {
    std::cout << "Unable to open file: " << filename << std::endl;
    return;
  }

  const int n_layer = layers.size();
  file.write(reinterpret_cast<const char *>(&n_layer), sizeof(n_layer));
  expected_file_size += sizeof(n_layer);

  std::vector<std::vector<float>> all_layers_params = this->get_parameters();

  for (const auto &layer_params : all_layers_params)
  {
    const int n_param = layer_params.size();
    file.write(reinterpret_cast<const char *>(&n_param), sizeof(n_param));

    if (n_param > 0)
      file.write(reinterpret_cast<const char *>(&layer_params[0]), sizeof(float) * n_param);

    expected_file_size += sizeof(n_param);
    expected_file_size += sizeof(float) * n_param;
  }

  std::streampos real_file_size = file.tellp();
  if (real_file_size != expected_file_size)
  {
    std::cout << "Expected file size: " << expected_file_size << " (bytes)" << std::endl;
    std::cout << "Real file size: " << real_file_size << " (bytes)" << std::endl;
    std::cerr << "The file size is not as expected, something may went wrong." << std::endl;
    return;
  }

  file.close();
  std::cout << "Parameters saved to \"" << filename << "\"." << std::endl;
}

void Network::load_parameters(const std::string &filename)
{
  std::ifstream file(filename, std::ios::binary | std::ios::in);

  if (file.is_open() == false)
  {
    std::cout << "Unable to open file: \"" << filename << "\" to load parameters." << std::endl;
    return;
  }

  int n_layer = 0;
  file.read(reinterpret_cast<char *>(&n_layer), sizeof(n_layer));

  std::vector<std::vector<float>> all_layers_params;
  all_layers_params.reserve(n_layer);

  for (int i = 0; i < n_layer; i++)
  {
    int n_param = 0;
    file.read(reinterpret_cast<char *>(&n_param), sizeof(n_param));

    std::vector<float> layer_params;
    layer_params.reserve(n_param);

    for (int j = 0; j < n_param; j++)
    {
      float param = 0;
      file.read(reinterpret_cast<char *>(&param), sizeof(param));
      layer_params.push_back(param);
    }

    all_layers_params.push_back(layer_params);
  }

  file.close();
  this->set_parameters(all_layers_params);
  std::cout << "Parameters loaded from \"" << filename << "\"." << std::endl;
}