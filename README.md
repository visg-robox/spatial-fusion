# spatial-fusion
## SPNet
>input:(batch_size, near_num, time_step, feature_dim)
>output:(batch_size, class_num)

### lstm:
得到query
>input:从当前batch的切片，near number中间那一维度
>output:qk_dim=256

### encoder
>input:(batch_size, near_num, time_step, feature_dim)
>output:(batch_size, near_num, time_step, qk_dim)

### attention
>input:encoder的output 和 lstm的output
attention
decoder
>output:(batch_size, class_num)

