// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: arch-state.proto

#include "arch-state.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
extern PROTOBUF_INTERNAL_EXPORT_arch_2dstate_2eproto ::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_CSR_arch_2dstate_2eproto;
class CSRDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<CSR> _instance;
} _CSR_default_instance_;
class ArchStateDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<ArchState> _instance;
} _ArchState_default_instance_;
static void InitDefaultsscc_info_ArchState_arch_2dstate_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::_ArchState_default_instance_;
    new (ptr) ::ArchState();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
  ::ArchState::InitAsDefaultInstance();
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<1> scc_info_ArchState_arch_2dstate_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 1, 0, InitDefaultsscc_info_ArchState_arch_2dstate_2eproto}, {
      &scc_info_CSR_arch_2dstate_2eproto.base,}};

static void InitDefaultsscc_info_CSR_arch_2dstate_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::_CSR_default_instance_;
    new (ptr) ::CSR();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
  ::CSR::InitAsDefaultInstance();
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_CSR_arch_2dstate_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 0, 0, InitDefaultsscc_info_CSR_arch_2dstate_2eproto}, {}};

static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_arch_2dstate_2eproto[2];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_arch_2dstate_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_arch_2dstate_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_arch_2dstate_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::CSR, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::CSR, msg_addr_),
  PROTOBUF_FIELD_OFFSET(::CSR, msg_csr_priv_),
  PROTOBUF_FIELD_OFFSET(::CSR, msg_csr_read_only_),
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::ArchState, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::ArchState, msg_pc_),
  PROTOBUF_FIELD_OFFSET(::ArchState, msg_mstatush_),
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::CSR)},
  { 8, -1, sizeof(::ArchState)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::_CSR_default_instance_),
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::_ArchState_default_instance_),
};

const char descriptor_table_protodef_arch_2dstate_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n\020arch-state.proto\"H\n\003CSR\022\020\n\010msg_addr\030\001 "
  "\001(\004\022\024\n\014msg_csr_priv\030\002 \001(\r\022\031\n\021msg_csr_rea"
  "d_only\030\003 \001(\010\"7\n\tArchState\022\016\n\006msg_pc\030\001 \001("
  "\004\022\032\n\014msg_mstatush\030\002 \001(\0132\004.CSRb\006proto3"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_arch_2dstate_2eproto_deps[1] = {
};
static ::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase*const descriptor_table_arch_2dstate_2eproto_sccs[2] = {
  &scc_info_ArchState_arch_2dstate_2eproto.base,
  &scc_info_CSR_arch_2dstate_2eproto.base,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_arch_2dstate_2eproto_once;
static bool descriptor_table_arch_2dstate_2eproto_initialized = false;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_arch_2dstate_2eproto = {
  &descriptor_table_arch_2dstate_2eproto_initialized, descriptor_table_protodef_arch_2dstate_2eproto, "arch-state.proto", 157,
  &descriptor_table_arch_2dstate_2eproto_once, descriptor_table_arch_2dstate_2eproto_sccs, descriptor_table_arch_2dstate_2eproto_deps, 2, 0,
  schemas, file_default_instances, TableStruct_arch_2dstate_2eproto::offsets,
  file_level_metadata_arch_2dstate_2eproto, 2, file_level_enum_descriptors_arch_2dstate_2eproto, file_level_service_descriptors_arch_2dstate_2eproto,
};

// Force running AddDescriptors() at dynamic initialization time.
static bool dynamic_init_dummy_arch_2dstate_2eproto = (  ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptors(&descriptor_table_arch_2dstate_2eproto), true);

// ===================================================================

void CSR::InitAsDefaultInstance() {
}
class CSR::_Internal {
 public:
};

CSR::CSR()
  : ::PROTOBUF_NAMESPACE_ID::Message(), _internal_metadata_(nullptr) {
  SharedCtor();
  // @@protoc_insertion_point(constructor:CSR)
}
CSR::CSR(const CSR& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _internal_metadata_(nullptr) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::memcpy(&msg_addr_, &from.msg_addr_,
    static_cast<size_t>(reinterpret_cast<char*>(&msg_csr_read_only_) -
    reinterpret_cast<char*>(&msg_addr_)) + sizeof(msg_csr_read_only_));
  // @@protoc_insertion_point(copy_constructor:CSR)
}

void CSR::SharedCtor() {
  ::memset(&msg_addr_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&msg_csr_read_only_) -
      reinterpret_cast<char*>(&msg_addr_)) + sizeof(msg_csr_read_only_));
}

CSR::~CSR() {
  // @@protoc_insertion_point(destructor:CSR)
  SharedDtor();
}

void CSR::SharedDtor() {
}

void CSR::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const CSR& CSR::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_CSR_arch_2dstate_2eproto.base);
  return *internal_default_instance();
}


void CSR::Clear() {
// @@protoc_insertion_point(message_clear_start:CSR)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  ::memset(&msg_addr_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&msg_csr_read_only_) -
      reinterpret_cast<char*>(&msg_addr_)) + sizeof(msg_csr_read_only_));
  _internal_metadata_.Clear();
}

const char* CSR::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // uint64 msg_addr = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8)) {
          msg_addr_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // uint32 msg_csr_priv = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 16)) {
          msg_csr_priv_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // bool msg_csr_read_only = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 24)) {
          msg_csr_read_only_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag & 7) == 4 || tag == 0) {
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag, &_internal_metadata_, ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* CSR::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:CSR)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // uint64 msg_addr = 1;
  if (this->msg_addr() != 0) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteUInt64ToArray(1, this->_internal_msg_addr(), target);
  }

  // uint32 msg_csr_priv = 2;
  if (this->msg_csr_priv() != 0) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteUInt32ToArray(2, this->_internal_msg_csr_priv(), target);
  }

  // bool msg_csr_read_only = 3;
  if (this->msg_csr_read_only() != 0) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(3, this->_internal_msg_csr_read_only(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields(), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:CSR)
  return target;
}

size_t CSR::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:CSR)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // uint64 msg_addr = 1;
  if (this->msg_addr() != 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::UInt64Size(
        this->_internal_msg_addr());
  }

  // uint32 msg_csr_priv = 2;
  if (this->msg_csr_priv() != 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::UInt32Size(
        this->_internal_msg_csr_priv());
  }

  // bool msg_csr_read_only = 3;
  if (this->msg_csr_read_only() != 0) {
    total_size += 1 + 1;
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void CSR::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:CSR)
  GOOGLE_DCHECK_NE(&from, this);
  const CSR* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<CSR>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:CSR)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:CSR)
    MergeFrom(*source);
  }
}

void CSR::MergeFrom(const CSR& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:CSR)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (from.msg_addr() != 0) {
    _internal_set_msg_addr(from._internal_msg_addr());
  }
  if (from.msg_csr_priv() != 0) {
    _internal_set_msg_csr_priv(from._internal_msg_csr_priv());
  }
  if (from.msg_csr_read_only() != 0) {
    _internal_set_msg_csr_read_only(from._internal_msg_csr_read_only());
  }
}

void CSR::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:CSR)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void CSR::CopyFrom(const CSR& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:CSR)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool CSR::IsInitialized() const {
  return true;
}

void CSR::InternalSwap(CSR* other) {
  using std::swap;
  _internal_metadata_.Swap(&other->_internal_metadata_);
  swap(msg_addr_, other->msg_addr_);
  swap(msg_csr_priv_, other->msg_csr_priv_);
  swap(msg_csr_read_only_, other->msg_csr_read_only_);
}

::PROTOBUF_NAMESPACE_ID::Metadata CSR::GetMetadata() const {
  return GetMetadataStatic();
}


// ===================================================================

void ArchState::InitAsDefaultInstance() {
  ::_ArchState_default_instance_._instance.get_mutable()->msg_mstatush_ = const_cast< ::CSR*>(
      ::CSR::internal_default_instance());
}
class ArchState::_Internal {
 public:
  static const ::CSR& msg_mstatush(const ArchState* msg);
};

const ::CSR&
ArchState::_Internal::msg_mstatush(const ArchState* msg) {
  return *msg->msg_mstatush_;
}
ArchState::ArchState()
  : ::PROTOBUF_NAMESPACE_ID::Message(), _internal_metadata_(nullptr) {
  SharedCtor();
  // @@protoc_insertion_point(constructor:ArchState)
}
ArchState::ArchState(const ArchState& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _internal_metadata_(nullptr) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  if (from._internal_has_msg_mstatush()) {
    msg_mstatush_ = new ::CSR(*from.msg_mstatush_);
  } else {
    msg_mstatush_ = nullptr;
  }
  msg_pc_ = from.msg_pc_;
  // @@protoc_insertion_point(copy_constructor:ArchState)
}

void ArchState::SharedCtor() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&scc_info_ArchState_arch_2dstate_2eproto.base);
  ::memset(&msg_mstatush_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&msg_pc_) -
      reinterpret_cast<char*>(&msg_mstatush_)) + sizeof(msg_pc_));
}

ArchState::~ArchState() {
  // @@protoc_insertion_point(destructor:ArchState)
  SharedDtor();
}

void ArchState::SharedDtor() {
  if (this != internal_default_instance()) delete msg_mstatush_;
}

void ArchState::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const ArchState& ArchState::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_ArchState_arch_2dstate_2eproto.base);
  return *internal_default_instance();
}


void ArchState::Clear() {
// @@protoc_insertion_point(message_clear_start:ArchState)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  if (GetArenaNoVirtual() == nullptr && msg_mstatush_ != nullptr) {
    delete msg_mstatush_;
  }
  msg_mstatush_ = nullptr;
  msg_pc_ = PROTOBUF_ULONGLONG(0);
  _internal_metadata_.Clear();
}

const char* ArchState::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // uint64 msg_pc = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8)) {
          msg_pc_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // .CSR msg_mstatush = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 18)) {
          ptr = ctx->ParseMessage(_internal_mutable_msg_mstatush(), ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag & 7) == 4 || tag == 0) {
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag, &_internal_metadata_, ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* ArchState::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:ArchState)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // uint64 msg_pc = 1;
  if (this->msg_pc() != 0) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteUInt64ToArray(1, this->_internal_msg_pc(), target);
  }

  // .CSR msg_mstatush = 2;
  if (this->has_msg_mstatush()) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(
        2, _Internal::msg_mstatush(this), target, stream);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields(), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:ArchState)
  return target;
}

size_t ArchState::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:ArchState)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // .CSR msg_mstatush = 2;
  if (this->has_msg_mstatush()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
        *msg_mstatush_);
  }

  // uint64 msg_pc = 1;
  if (this->msg_pc() != 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::UInt64Size(
        this->_internal_msg_pc());
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void ArchState::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:ArchState)
  GOOGLE_DCHECK_NE(&from, this);
  const ArchState* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<ArchState>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:ArchState)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:ArchState)
    MergeFrom(*source);
  }
}

void ArchState::MergeFrom(const ArchState& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:ArchState)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (from.has_msg_mstatush()) {
    _internal_mutable_msg_mstatush()->::CSR::MergeFrom(from._internal_msg_mstatush());
  }
  if (from.msg_pc() != 0) {
    _internal_set_msg_pc(from._internal_msg_pc());
  }
}

void ArchState::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:ArchState)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void ArchState::CopyFrom(const ArchState& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:ArchState)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool ArchState::IsInitialized() const {
  return true;
}

void ArchState::InternalSwap(ArchState* other) {
  using std::swap;
  _internal_metadata_.Swap(&other->_internal_metadata_);
  swap(msg_mstatush_, other->msg_mstatush_);
  swap(msg_pc_, other->msg_pc_);
}

::PROTOBUF_NAMESPACE_ID::Metadata ArchState::GetMetadata() const {
  return GetMetadataStatic();
}


// @@protoc_insertion_point(namespace_scope)
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::CSR* Arena::CreateMaybeMessage< ::CSR >(Arena* arena) {
  return Arena::CreateInternal< ::CSR >(arena);
}
template<> PROTOBUF_NOINLINE ::ArchState* Arena::CreateMaybeMessage< ::ArchState >(Arena* arena) {
  return Arena::CreateInternal< ::ArchState >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
