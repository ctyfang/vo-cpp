#include <iostream>
#include <boost/program_options.hpp>

#include "pipeline.h"

namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "Produce help message")
        ("param", po::value<std::string>(), "Path to YAML parameter file");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    std::shared_ptr<Pipeline> pipeline_ptr;
    if (vm.count("param")) {
        std::cout << "Parameter filepath: " << vm["param"].as<std::string>() << ".\n";
        cv::FileStorage param_node(vm["param"].as<std::string>(), cv::FileStorage::READ);
        pipeline_ptr = std::make_shared<Pipeline>(param_node);

        pipeline_ptr->Initialize();
        while(true) {
            pipeline_ptr->Update();
        }
    } else {
        std::cout << "Path to parameter file was not set.\n";
        return 1;
    }

    
    return 0;
}
