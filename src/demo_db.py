"""Synthetic example databases (in-memory SQLite) for the UI demo. All data is fictional."""
from __future__ import annotations

import random
import sqlite3

DEMO_DATABASES: dict[str, dict] = {
    "Online Store": {
        "schema": (
            "CREATE TABLE customers (customer_id INTEGER, name VARCHAR, city VARCHAR); "
            "CREATE TABLE products (product_id INTEGER, product_name VARCHAR, category VARCHAR, price REAL); "
            "CREATE TABLE orders (order_id INTEGER, customer_id INTEGER, product_id INTEGER, quantity INTEGER, order_date VARCHAR)"
        ),
        "examples": [
            "How many customers are from Jakarta?",
            "What is the average price of products in each category?",
            "List the names of customers who ordered more than 3 items in one order.",
        ],
    },
    "Campus": {
        "schema": (
            "CREATE TABLE students (student_id INTEGER, name VARCHAR, major VARCHAR, gpa REAL); "
            "CREATE TABLE courses (course_id INTEGER, title VARCHAR, credits INTEGER); "
            "CREATE TABLE enrollments (student_id INTEGER, course_id INTEGER, grade VARCHAR)"
        ),
        "examples": [
            "How many students are in each major?",
            "List the names of students with a GPA above 3.5.",
            "Which courses have more than 2 credits?",
        ],
    },
    "Hospital": {
        "schema": (
            "CREATE TABLE patients (patient_id INTEGER, name VARCHAR, age INTEGER, city VARCHAR); "
            "CREATE TABLE doctors (doctor_id INTEGER, name VARCHAR, specialty VARCHAR); "
            "CREATE TABLE appointments (appointment_id INTEGER, patient_id INTEGER, doctor_id INTEGER, visit_date VARCHAR, fee REAL)"
        ),
        "examples": [
            "What is the average age of patients?",
            "How many appointments does each doctor have?",
            "List the names of doctors whose specialty is Cardiology.",
        ],
    },
}

_NAMES = ["Budi", "Siti", "Agus", "Dewi", "Rizky", "Putri", "Andi", "Maya", "Fajar", "Lestari", "Hendra", "Nadia"]
_CITIES = ["Jakarta", "Surabaya", "Bandung", "Medan", "Semarang"]


def list_databases() -> list[str]:
    return list(DEMO_DATABASES)


def get_schema(name: str) -> str:
    return DEMO_DATABASES[name]["schema"]


def get_examples(name: str) -> list[str]:
    return DEMO_DATABASES[name]["examples"]


def create_demo_connection(name: str, seed: int = 42) -> sqlite3.Connection:
    """Build an in-memory demo DB (schema + deterministic fictional data)."""
    rng = random.Random(seed)
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    for stmt in get_schema(name).split(";"):
        if stmt.strip():
            conn.execute(stmt)
    if name == "Online Store":
        conn.executemany("INSERT INTO customers VALUES (?,?,?)", [(i, _NAMES[i % 12], rng.choice(_CITIES)) for i in range(1, 21)])
        cats = ["Electronics", "Fashion", "Books", "Home"]
        conn.executemany("INSERT INTO products VALUES (?,?,?,?)", [(i, f"Product {i}", cats[i % 4], round(rng.uniform(10, 500), 1)) for i in range(1, 21)])
        conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", [
            (i, rng.randint(1, 20), rng.randint(1, 20), rng.randint(1, 6), f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}") for i in range(1, 61)])
    elif name == "Campus":
        majors = ["Physics", "Informatics", "Biology", "Mathematics"]
        conn.executemany("INSERT INTO students VALUES (?,?,?,?)", [(i, _NAMES[i % 12], majors[i % 4], round(rng.uniform(2.0, 4.0), 2)) for i in range(1, 31)])
        conn.executemany("INSERT INTO courses VALUES (?,?,?)", [(i, f"Course {i}", rng.randint(1, 4)) for i in range(1, 11)])
        conn.executemany("INSERT INTO enrollments VALUES (?,?,?)", [(rng.randint(1, 30), rng.randint(1, 10), rng.choice(["A", "B", "C"])) for _ in range(60)])
    elif name == "Hospital":
        conn.executemany("INSERT INTO patients VALUES (?,?,?,?)", [(i, _NAMES[i % 12], rng.randint(5, 80), rng.choice(_CITIES)) for i in range(1, 31)])
        specs = ["Cardiology", "Pediatrics", "Neurology", "Dermatology"]
        conn.executemany("INSERT INTO doctors VALUES (?,?,?)", [(i, f"Dr. {_NAMES[i % 12]}", specs[i % 4]) for i in range(1, 9)])
        conn.executemany("INSERT INTO appointments VALUES (?,?,?,?,?)", [
            (i, rng.randint(1, 30), rng.randint(1, 8), f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}", round(rng.uniform(50, 300), 1)) for i in range(1, 51)])
    conn.commit()
    return conn
