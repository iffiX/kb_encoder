if __name__ == "__main__":
    no_match_correct_lines = []
    match_all_correct_lines = []
    match_specific_correct_lines = []
    no_match_correct = []
    match_all_correct = []
    match_specific_correct = []
    with open("train_t5_best_67.3.txt", "r") as nm, open(
        "train_t5_match_all_anno_best_72.6.txt"
    ) as ma, open("train_t5_match_approx_best_90.5.txt") as ms:
        ans_begin_len = len("answer: [")
        ref_begin_len = len("ref_answer: [")
        for i in range(1221):
            q, nm_ans, nm_ref = nm.readline(), nm.readline(), nm.readline()
            q1, ma_ans, ma_ref = ma.readline(), ma.readline(), ma.readline()
            q2, ms_ans, ms_ref = ms.readline(), ms.readline(), ms.readline()
            nm_ans = nm_ans[ans_begin_len : nm_ans.find("]")]
            nm_ref = nm_ref[ref_begin_len : nm_ref.find("]")]
            ma_ans = ma_ans[ans_begin_len : ma_ans.find("]")]
            ma_ref = ma_ref[ref_begin_len : ma_ref.find("]")]
            ms_ans = ms_ans[ans_begin_len : ms_ans.find("]")]
            ms_ref = ms_ref[ref_begin_len : ms_ref.find("]")]

            no_match_correct.append(q)
            no_match_correct.append(nm_ans)
            no_match_correct.append(nm_ref)

            match_all_correct.append(q1)
            match_all_correct.append(ma_ans)
            match_all_correct.append(ma_ref)

            match_specific_correct.append(q2)
            match_specific_correct.append(ms_ans)
            match_specific_correct.append(ms_ref)

            if nm_ans == nm_ref:
                no_match_correct_lines.append(i)
            if ma_ans == ma_ref:
                match_all_correct_lines.append(i)
            if ms_ans == ms_ref:
                match_specific_correct_lines.append(i)

    all_correct = (
        len(
            set(no_match_correct_lines)
            .intersection(set(match_all_correct_lines))
            .intersection(set(match_specific_correct_lines))
        )
        / 1221
    )
    print(f"same correct: {all_correct * 100:.2f}%")
    nm_ma_correct = (
        len(set(no_match_correct_lines).intersection(set(match_all_correct_lines)))
        / 1221
    )
    print(f"nm and ma correct: {nm_ma_correct * 100:.2f}%")
    nm_ms_correct = (
        len(set(no_match_correct_lines).intersection(set(match_specific_correct_lines)))
        / 1221
    )
    print(f"nm and ms correct: {nm_ms_correct * 100:.2f}%")
    ma_ms_correct = (
        len(
            set(match_all_correct_lines).intersection(set(match_specific_correct_lines))
        )
        / 1221
    )
    print(f"ma and ms correct: {ma_ms_correct * 100:.2f}%")

    for i in set(match_specific_correct_lines).difference(
        no_match_correct_lines, match_all_correct_lines
    ):
        print(f"question: {match_specific_correct[i*3]}")
        print(f"answer: {match_specific_correct[i * 3 + 1]}")
