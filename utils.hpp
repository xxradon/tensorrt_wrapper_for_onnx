#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include <cuda_runtime_api.h>
#include <iostream>
#include "assert.h"


#define CHECK_ASSERT(x, format, args...) do {   \
    if(!(x)) {                                    \
        printf(format, ##args);                 \
        assert(0);                              \
    }                                           \
} while(0)

#define LOG(format, args...) do {               \
        printf(format, ##args);                 \
} while(0)


namespace tensorrtInference
{

    class Logger : public nvinfer1::ILogger
    {
    public:

        Logger(): Logger(Severity::kWARNING) {}

        Logger(Severity severity): reportableSeverity(severity) {}

        void log(Severity severity, const char* msg) override
        {
            // suppress messages with severity enum value greater than the reportable
            if (severity > reportableSeverity) return;

            switch (severity)
            {
            case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
            case Severity::kERROR: std::cerr << "ERROR: "; break;
            case Severity::kWARNING: std::cerr << "WARNING: "; break;
            case Severity::kINFO: std::cerr << "INFO: "; break;
            default: std::cerr << "UNKNOWN: "; break;
            }
            std::cerr << msg << std::endl;
        }

        Severity reportableSeverity{Severity::kWARNING};
    };

}
#endif