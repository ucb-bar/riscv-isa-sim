// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: arch-state.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_arch_2dstate_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_arch_2dstate_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3011000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3011002 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/inlined_string_field.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_arch_2dstate_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_arch_2dstate_2eproto {
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::AuxillaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTable schema[2]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::FieldMetadata field_metadata[];
  static const ::PROTOBUF_NAMESPACE_ID::internal::SerializationTable serialization_table[];
  static const ::PROTOBUF_NAMESPACE_ID::uint32 offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_arch_2dstate_2eproto;
class ArchState;
class ArchStateDefaultTypeInternal;
extern ArchStateDefaultTypeInternal _ArchState_default_instance_;
class CSR;
class CSRDefaultTypeInternal;
extern CSRDefaultTypeInternal _CSR_default_instance_;
PROTOBUF_NAMESPACE_OPEN
template<> ::ArchState* Arena::CreateMaybeMessage<::ArchState>(Arena*);
template<> ::CSR* Arena::CreateMaybeMessage<::CSR>(Arena*);
PROTOBUF_NAMESPACE_CLOSE

// ===================================================================

class CSR :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:CSR) */ {
 public:
  CSR();
  virtual ~CSR();

  CSR(const CSR& from);
  CSR(CSR&& from) noexcept
    : CSR() {
    *this = ::std::move(from);
  }

  inline CSR& operator=(const CSR& from) {
    CopyFrom(from);
    return *this;
  }
  inline CSR& operator=(CSR&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return GetMetadataStatic().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return GetMetadataStatic().reflection;
  }
  static const CSR& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const CSR* internal_default_instance() {
    return reinterpret_cast<const CSR*>(
               &_CSR_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(CSR& a, CSR& b) {
    a.Swap(&b);
  }
  inline void Swap(CSR* other) {
    if (other == this) return;
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline CSR* New() const final {
    return CreateMaybeMessage<CSR>(nullptr);
  }

  CSR* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<CSR>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const CSR& from);
  void MergeFrom(const CSR& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  ::PROTOBUF_NAMESPACE_ID::uint8* _InternalSerialize(
      ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(CSR* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "CSR";
  }
  private:
  inline ::PROTOBUF_NAMESPACE_ID::Arena* GetArenaNoVirtual() const {
    return nullptr;
  }
  inline void* MaybeArenaPtr() const {
    return nullptr;
  }
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_arch_2dstate_2eproto);
    return ::descriptor_table_arch_2dstate_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kMsgAddrFieldNumber = 1,
    kMsgCsrPrivFieldNumber = 2,
    kMsgCsrReadOnlyFieldNumber = 3,
  };
  // uint64 msg_addr = 1;
  void clear_msg_addr();
  ::PROTOBUF_NAMESPACE_ID::uint64 msg_addr() const;
  void set_msg_addr(::PROTOBUF_NAMESPACE_ID::uint64 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::uint64 _internal_msg_addr() const;
  void _internal_set_msg_addr(::PROTOBUF_NAMESPACE_ID::uint64 value);
  public:

  // uint32 msg_csr_priv = 2;
  void clear_msg_csr_priv();
  ::PROTOBUF_NAMESPACE_ID::uint32 msg_csr_priv() const;
  void set_msg_csr_priv(::PROTOBUF_NAMESPACE_ID::uint32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::uint32 _internal_msg_csr_priv() const;
  void _internal_set_msg_csr_priv(::PROTOBUF_NAMESPACE_ID::uint32 value);
  public:

  // bool msg_csr_read_only = 3;
  void clear_msg_csr_read_only();
  bool msg_csr_read_only() const;
  void set_msg_csr_read_only(bool value);
  private:
  bool _internal_msg_csr_read_only() const;
  void _internal_set_msg_csr_read_only(bool value);
  public:

  // @@protoc_insertion_point(class_scope:CSR)
 private:
  class _Internal;

  ::PROTOBUF_NAMESPACE_ID::internal::InternalMetadataWithArena _internal_metadata_;
  ::PROTOBUF_NAMESPACE_ID::uint64 msg_addr_;
  ::PROTOBUF_NAMESPACE_ID::uint32 msg_csr_priv_;
  bool msg_csr_read_only_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_arch_2dstate_2eproto;
};
// -------------------------------------------------------------------

class ArchState :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:ArchState) */ {
 public:
  ArchState();
  virtual ~ArchState();

  ArchState(const ArchState& from);
  ArchState(ArchState&& from) noexcept
    : ArchState() {
    *this = ::std::move(from);
  }

  inline ArchState& operator=(const ArchState& from) {
    CopyFrom(from);
    return *this;
  }
  inline ArchState& operator=(ArchState&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return GetMetadataStatic().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return GetMetadataStatic().reflection;
  }
  static const ArchState& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const ArchState* internal_default_instance() {
    return reinterpret_cast<const ArchState*>(
               &_ArchState_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  friend void swap(ArchState& a, ArchState& b) {
    a.Swap(&b);
  }
  inline void Swap(ArchState* other) {
    if (other == this) return;
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline ArchState* New() const final {
    return CreateMaybeMessage<ArchState>(nullptr);
  }

  ArchState* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<ArchState>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const ArchState& from);
  void MergeFrom(const ArchState& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  ::PROTOBUF_NAMESPACE_ID::uint8* _InternalSerialize(
      ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(ArchState* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "ArchState";
  }
  private:
  inline ::PROTOBUF_NAMESPACE_ID::Arena* GetArenaNoVirtual() const {
    return nullptr;
  }
  inline void* MaybeArenaPtr() const {
    return nullptr;
  }
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_arch_2dstate_2eproto);
    return ::descriptor_table_arch_2dstate_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kMsgMstatushFieldNumber = 2,
    kMsgPcFieldNumber = 1,
  };
  // .CSR msg_mstatush = 2;
  bool has_msg_mstatush() const;
  private:
  bool _internal_has_msg_mstatush() const;
  public:
  void clear_msg_mstatush();
  const ::CSR& msg_mstatush() const;
  ::CSR* release_msg_mstatush();
  ::CSR* mutable_msg_mstatush();
  void set_allocated_msg_mstatush(::CSR* msg_mstatush);
  private:
  const ::CSR& _internal_msg_mstatush() const;
  ::CSR* _internal_mutable_msg_mstatush();
  public:

  // uint64 msg_pc = 1;
  void clear_msg_pc();
  ::PROTOBUF_NAMESPACE_ID::uint64 msg_pc() const;
  void set_msg_pc(::PROTOBUF_NAMESPACE_ID::uint64 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::uint64 _internal_msg_pc() const;
  void _internal_set_msg_pc(::PROTOBUF_NAMESPACE_ID::uint64 value);
  public:

  // @@protoc_insertion_point(class_scope:ArchState)
 private:
  class _Internal;

  ::PROTOBUF_NAMESPACE_ID::internal::InternalMetadataWithArena _internal_metadata_;
  ::CSR* msg_mstatush_;
  ::PROTOBUF_NAMESPACE_ID::uint64 msg_pc_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_arch_2dstate_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// CSR

// uint64 msg_addr = 1;
inline void CSR::clear_msg_addr() {
  msg_addr_ = PROTOBUF_ULONGLONG(0);
}
inline ::PROTOBUF_NAMESPACE_ID::uint64 CSR::_internal_msg_addr() const {
  return msg_addr_;
}
inline ::PROTOBUF_NAMESPACE_ID::uint64 CSR::msg_addr() const {
  // @@protoc_insertion_point(field_get:CSR.msg_addr)
  return _internal_msg_addr();
}
inline void CSR::_internal_set_msg_addr(::PROTOBUF_NAMESPACE_ID::uint64 value) {
  
  msg_addr_ = value;
}
inline void CSR::set_msg_addr(::PROTOBUF_NAMESPACE_ID::uint64 value) {
  _internal_set_msg_addr(value);
  // @@protoc_insertion_point(field_set:CSR.msg_addr)
}

// uint32 msg_csr_priv = 2;
inline void CSR::clear_msg_csr_priv() {
  msg_csr_priv_ = 0u;
}
inline ::PROTOBUF_NAMESPACE_ID::uint32 CSR::_internal_msg_csr_priv() const {
  return msg_csr_priv_;
}
inline ::PROTOBUF_NAMESPACE_ID::uint32 CSR::msg_csr_priv() const {
  // @@protoc_insertion_point(field_get:CSR.msg_csr_priv)
  return _internal_msg_csr_priv();
}
inline void CSR::_internal_set_msg_csr_priv(::PROTOBUF_NAMESPACE_ID::uint32 value) {
  
  msg_csr_priv_ = value;
}
inline void CSR::set_msg_csr_priv(::PROTOBUF_NAMESPACE_ID::uint32 value) {
  _internal_set_msg_csr_priv(value);
  // @@protoc_insertion_point(field_set:CSR.msg_csr_priv)
}

// bool msg_csr_read_only = 3;
inline void CSR::clear_msg_csr_read_only() {
  msg_csr_read_only_ = false;
}
inline bool CSR::_internal_msg_csr_read_only() const {
  return msg_csr_read_only_;
}
inline bool CSR::msg_csr_read_only() const {
  // @@protoc_insertion_point(field_get:CSR.msg_csr_read_only)
  return _internal_msg_csr_read_only();
}
inline void CSR::_internal_set_msg_csr_read_only(bool value) {
  
  msg_csr_read_only_ = value;
}
inline void CSR::set_msg_csr_read_only(bool value) {
  _internal_set_msg_csr_read_only(value);
  // @@protoc_insertion_point(field_set:CSR.msg_csr_read_only)
}

// -------------------------------------------------------------------

// ArchState

// uint64 msg_pc = 1;
inline void ArchState::clear_msg_pc() {
  msg_pc_ = PROTOBUF_ULONGLONG(0);
}
inline ::PROTOBUF_NAMESPACE_ID::uint64 ArchState::_internal_msg_pc() const {
  return msg_pc_;
}
inline ::PROTOBUF_NAMESPACE_ID::uint64 ArchState::msg_pc() const {
  // @@protoc_insertion_point(field_get:ArchState.msg_pc)
  return _internal_msg_pc();
}
inline void ArchState::_internal_set_msg_pc(::PROTOBUF_NAMESPACE_ID::uint64 value) {
  
  msg_pc_ = value;
}
inline void ArchState::set_msg_pc(::PROTOBUF_NAMESPACE_ID::uint64 value) {
  _internal_set_msg_pc(value);
  // @@protoc_insertion_point(field_set:ArchState.msg_pc)
}

// .CSR msg_mstatush = 2;
inline bool ArchState::_internal_has_msg_mstatush() const {
  return this != internal_default_instance() && msg_mstatush_ != nullptr;
}
inline bool ArchState::has_msg_mstatush() const {
  return _internal_has_msg_mstatush();
}
inline void ArchState::clear_msg_mstatush() {
  if (GetArenaNoVirtual() == nullptr && msg_mstatush_ != nullptr) {
    delete msg_mstatush_;
  }
  msg_mstatush_ = nullptr;
}
inline const ::CSR& ArchState::_internal_msg_mstatush() const {
  const ::CSR* p = msg_mstatush_;
  return p != nullptr ? *p : *reinterpret_cast<const ::CSR*>(
      &::_CSR_default_instance_);
}
inline const ::CSR& ArchState::msg_mstatush() const {
  // @@protoc_insertion_point(field_get:ArchState.msg_mstatush)
  return _internal_msg_mstatush();
}
inline ::CSR* ArchState::release_msg_mstatush() {
  // @@protoc_insertion_point(field_release:ArchState.msg_mstatush)
  
  ::CSR* temp = msg_mstatush_;
  msg_mstatush_ = nullptr;
  return temp;
}
inline ::CSR* ArchState::_internal_mutable_msg_mstatush() {
  
  if (msg_mstatush_ == nullptr) {
    auto* p = CreateMaybeMessage<::CSR>(GetArenaNoVirtual());
    msg_mstatush_ = p;
  }
  return msg_mstatush_;
}
inline ::CSR* ArchState::mutable_msg_mstatush() {
  // @@protoc_insertion_point(field_mutable:ArchState.msg_mstatush)
  return _internal_mutable_msg_mstatush();
}
inline void ArchState::set_allocated_msg_mstatush(::CSR* msg_mstatush) {
  ::PROTOBUF_NAMESPACE_ID::Arena* message_arena = GetArenaNoVirtual();
  if (message_arena == nullptr) {
    delete msg_mstatush_;
  }
  if (msg_mstatush) {
    ::PROTOBUF_NAMESPACE_ID::Arena* submessage_arena = nullptr;
    if (message_arena != submessage_arena) {
      msg_mstatush = ::PROTOBUF_NAMESPACE_ID::internal::GetOwnedMessage(
          message_arena, msg_mstatush, submessage_arena);
    }
    
  } else {
    
  }
  msg_mstatush_ = msg_mstatush;
  // @@protoc_insertion_point(field_set_allocated:ArchState.msg_mstatush)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)


// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_arch_2dstate_2eproto
