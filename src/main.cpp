#include <iostream>
#include <boost/program_options.hpp>


namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "Produce help message")
        ("source", po::value<std::string>(), "Source of data ['KITTI', 'camera']"); 
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    if (vm.count("source")) {
        std::cout << "Specified source: " << vm["source"].as<std::string>() << ".\n";
    } else {
        std::cout << "Data source was not set.\n";
    }

    return 0;
}
