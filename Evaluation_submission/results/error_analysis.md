# Error Analysis - fine-tuned model

The examples below are taken automatically from real predictions; the 'Likely cause' line is a heuristic, not a manual judgment.

## simple (30 failed out of 217)

**Question:** What score in the final had a tournament of $25,000 Glasgow, Great Britain?
- Schema: `CREATE TABLE table_name_82 (score_in_the_final VARCHAR, tournament VARCHAR)`
- Reference: `select score_in_the_final from table_name_82 where tournament='$25,000 glasgow,great britain'`
- Prediction: `select score_in_the_final from table_name_82 where tournament='$25,000 gosport,great britain'`
- Likely cause: Similar structure, but the columns/values/aggregation differ from the reference.

**Question:** 4tab of commercial broadcaster Radio Tab broadcasts at what frequency?
- Schema: `CREATE TABLE table_name_75 (frequency VARCHAR, callsign VARCHAR, purpose VARCHAR, on_air_id VARCHAR)`
- Reference: `select frequency from table_name_75 where purpose='commercial' and on_air_id='radio tab' and callsign='4tab'`
- Prediction: `select frequency from table_name_75 where purpose='commercial broadcaster' and on_air_id='radio tab'`
- Likely cause: Similar structure, but the columns/values/aggregation differ from the reference.

**Question:** How many career titles does roy emerson have?
- Schema: `CREATE TABLE table_23408094_14 (total_career_titles VARCHAR, player VARCHAR)`
- Reference: `select total_career_titles from table_23408094_14 where player='roy emerson'`
- Prediction: `select count(total_career_titles) from table_23408094_14 where player='roy emerson'`
- Likely cause: Similar structure, but the columns/values/aggregation differ from the reference.

## aggregation (10 failed out of 79)

**Question:** Show all card type codes and the number of customers holding cards in each type.
- Schema: `CREATE TABLE Customers_cards (card_type_code VARCHAR, customer_id VARCHAR)`
- Reference: `select card_type_code,count(distinct customer_id) from customers_cards group by card_type_code`
- Prediction: `select card_type_code,count(*) from customers_cards group by card_type_code`
- Likely cause: Similar structure, but the columns/values/aggregation differ from the reference.

**Question:** How many teams had a #1 draft pick that won the Rookie of the Year Award?
- Schema: `CREATE TABLE table_1004033_1 (team VARCHAR, draft_pick__number VARCHAR)`
- Reference: `select count(team) from table_1004033_1 where draft_pick__number='1'`
- Prediction: `select count(team) from table_1004033_1 where draft_pick__number=1 and 'rookie of the year'='won'`
- Likely cause: Similar structure, but the columns/values/aggregation differ from the reference.

**Question:** What is the last year total for the team with a lowest of 23578?
- Schema: `CREATE TABLE table_2472711_31 (last_year INTEGER, lowest VARCHAR)`
- Reference: `select max(last_year) from table_2472711_31 where lowest=23578`
- Prediction: `select sum(last_year) from table_2472711_31 where lowest=23578`
- Likely cause: Similar structure, but the columns/values/aggregation differ from the reference.

## join (1 failed out of 4)

**Question:** What are the different ids and names of the battles that lost any 'Brig' type shipes?
- Schema: `CREATE TABLE ship (lost_in_battle VARCHAR, ship_type VARCHAR); CREATE TABLE battle (id VARCHAR, name VARCHAR)`
- Reference: `select distinct t1.id,t1.name from battle as t1 join ship as t2 on t1.id=t2.lost_in_battle where t2.ship_type='brig'`
- Prediction: `select t2.id,t2.name from ship as t1 join battle as t2 on t1.lost_in_battle=t2.name where t1.ship_type='brig'`
- Likely cause: Similar structure, but the columns/values/aggregation differ from the reference.

## nested (0 failed out of 0)

_No failures in this category._
