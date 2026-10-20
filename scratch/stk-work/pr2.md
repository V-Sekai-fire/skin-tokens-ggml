Vendors skin-tokens-ggml with the robust weight transfer (on by default).

- `robust=0` restores the upstream interpolation; the rig is byte-identical to the previous ELF (checked on SK_AnimatedWoman).
- `trace=1` keeps the transfer inputs for `skintokens_trace_size` / `skintokens_trace_chunk`, read by `skintokens-transfer-replay`.
- The job log gains a transfer line (matched, inpainted, trimmed parts, solve).

All 11 test models rig on Metal with 0 torn vertices in `skintokens-skin-stress` at seeds 1-3, 60 and 90 degrees.
