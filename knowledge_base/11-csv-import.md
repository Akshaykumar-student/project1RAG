# Import customers from CSV

Admins can import customers from **Customers → Import CSV**. The file must be UTF-8 encoded, no larger than 10 MB, and contain at most 25,000 rows. An `email` column is required. Optional columns include `name`, `company`, `phone`, and up to 20 custom attributes.

Existing customers are matched by lowercase email address and updated rather than duplicated. FlowDesk provides an error report for skipped rows. Imports cannot be undone, so export customers before a large update.

