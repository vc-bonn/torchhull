#include <torchhull/io.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace torchhull
{

void
store_curve_network_obj(const std::string& filename,
                        const std::tuple<torch::Tensor, torch::Tensor>& curve_network,
                        const bool verbose)
{
    if (verbose)
    {
        std::cout << "Storing file \"" + filename + "\" ..." << std::endl;
    }
    auto t_start = std::chrono::high_resolution_clock::now();

    std::ofstream file;
    file.open(filename);

    auto [verts, edges] = curve_network;

    // Vertices
    auto host_verts = verts.to(torch::kCPU);
    for (int j = 0; j < host_verts.size(0); ++j)
    {
        file << "v " << host_verts[j][0].item<float>() << " " << host_verts[j][1].item<float>() << " "
             << host_verts[j][2].item<float>() << "\n";
    }

    // Edges
    auto host_edges = edges.to(torch::kCPU);
    for (int j = 0; j < edges.size(0); ++j)
    {
        // 1-indexing
        file << "l " << 1 + host_edges[j][0].item<int64_t>() << " " << 1 + host_edges[j][1].item<int64_t>() << "\n";
    }

    file.close();

    auto t_end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start);
    if (verbose)
    {
        std::cout << "Storing file \"" + filename + "\" ... done (" << static_cast<float>(time.count()) / 1000.f << "s)"
                  << std::endl;
    }
}

void
store_curve_network_ply(const std::string& filename,
                        const std::tuple<torch::Tensor, torch::Tensor>& curve_network,
                        const bool verbose)
{
    if (verbose)
    {
        std::cout << "Storing file \"" + filename + "\" ..." << std::endl;
    }
    auto t_start = std::chrono::high_resolution_clock::now();

    auto [verts, edges] = curve_network;

    auto total_verts = verts.size(0);
    auto total_edges = edges.size(0);

    auto file = std::ofstream{};
    file.open(filename);

    file << "ply\n";
    file << "format binary_little_endian 1.0\n";

    file << "element vertex " << total_verts << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";

    file << "element edge " << total_edges << "\n";
    file << "property int vertex1\n";
    file << "property int vertex2\n";

    file << "end_header\n";

    file.close();

    auto file_binary = std::ofstream(filename, std::ios::app | std::ios::binary);

    // Vertices
    auto host_verts = verts.to(torch::kCPU).contiguous();
    file_binary.write(reinterpret_cast<const char*>(host_verts.data_ptr()), host_verts.numel() * sizeof(float));

    // Edges
    auto host_edges = edges.to(torch::kCPU).to(torch::kInt).contiguous().clone();
    file_binary.write(reinterpret_cast<const char*>(host_edges.data_ptr()), host_edges.numel() * sizeof(int));

    file_binary.close();

    auto t_end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start);
    if (verbose)
    {
        std::cout << "Storing file \"" + filename + "\" ... done (" << static_cast<float>(time.count()) / 1000.f << "s)"
                  << std::endl;
    }
}

void
store_curve_network(const std::string& filename,
                    const std::tuple<torch::Tensor, torch::Tensor>& curve_network,
                    const bool verbose)
{
    auto filename_path = std::filesystem::path(filename);

    if (filename_path.extension() == ".obj")
    {
        return store_curve_network_obj(filename, curve_network, verbose);
    }
    else if (filename_path.extension() == ".ply")
    {
        return store_curve_network_ply(filename, curve_network, verbose);
    }

    TORCH_CHECK(false, "No implementation for file extension \"" + filename_path.extension().string() + "\".");
}

} // namespace torchhull
