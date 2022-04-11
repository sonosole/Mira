kbytesof(b::B) where B <: Block = bytesof(b, "KB")
mbytesof(b::B) where B <: Block = bytesof(b, "MB")
gbytesof(b::B) where B <: Block = bytesof(b, "GB")
tbytesof(b::B) where B <: Block = bytesof(b, "TB")

