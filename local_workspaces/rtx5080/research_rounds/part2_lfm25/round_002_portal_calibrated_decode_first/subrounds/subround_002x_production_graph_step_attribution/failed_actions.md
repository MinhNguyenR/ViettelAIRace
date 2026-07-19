# R002x Failed Actions

## Invalid `.84` Startup

The first control-A process stopped before model load because desktop use left
only `12.63/15.44 GiB` free while `.84` requested `12.97 GiB`. No panel was
armed and no timing was retained. R002x uses matched `.80` for all three short
decode-profile processes, consistent with prior R002n-R002r profiling. This is
not a serving-config promotion.
