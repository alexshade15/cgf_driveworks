#include <dwcgf/channel/ChannelFactory.hpp>
#include <dwcgf/channel/IChannelPacket.hpp>

#include "YoloNet.hpp"

using IChannelPacket = dw::framework::IChannelPacket;
using GenericData = dw::framework::GenericData;

class YoloPacket : public IChannelPacket {
 public:
  YoloPacket(GenericData specimen) {
    auto* size = specimen.getData<size_t>();
    if (size != nullptr) {
      m_dataBuffer = std::make_unique<YoloScoreRect[]>(*size);
      m_packet.data = static_cast<YoloScoreRect*>(m_dataBuffer.get());
    } else {
      throw std::runtime_error("Invalid maximum buffer size.");
    }
    m_maxDataSize = *size;
  }
  GenericData getGenericData() override { return GenericData(&m_packet); }

 protected:
  YoloScoreRectArray m_packet{};
  std::unique_ptr<YoloScoreRect[]> m_dataBuffer{};
  size_t m_maxDataSize;
};

namespace {
struct Proxy {
  using ChannelPacketConstructor = dw::framework::ChannelPacketConstructor;
  using ChannelPacketConstructorSignature =
      dw::framework::ChannelPacketConstructorSignature;
  using ChannelFactory = dw::framework::ChannelFactory;

  Proxy() {
    m_sigShem = {YoloTypeID, dw::framework::ChannelType::SHMEM_LOCAL};

    ChannelFactory::registerPacketConstructor(
        m_sigShem,
        ChannelPacketConstructor([](GenericData ref, dwContextHandle_t context)
                                     -> std::unique_ptr<IChannelPacket> {
          static_cast<void>(context);
          return std::make_unique<YoloPacket>(ref);
        }));
  }
  ~Proxy() { ChannelFactory::unregisterPacketConstructor(m_sigShem); }

 private:
  ChannelPacketConstructorSignature m_sigShem;
};
static Proxy g_registerPacketConstructors;
}  // namespace
