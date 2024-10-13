"use client"

import { BusinessExplore } from "@/app/modules/business/BusinessExplore";
import Toolbar from "@/app/modules/Toolbar";
import { Business } from "@/types/BusinessTypes";
import { Metadata } from "next";
import { useState, useEffect } from "react";
import { database } from "@../../../firebaseConfig";
import { ref, set, get, child } from "firebase/database";


export default function forYouPage() {
    const [businesses, setBusinesses] = useState<Business[]>([]);

    const readUnseenBusinesses = async () => {
        const dbRef = ref(database);
        const snapshot = await get(child(dbRef, 'unseen'));
        if (snapshot.exists()) {
            return snapshot.val();
        } else {
            console.log("No rejected businesses available");
            return [];
        }
    };

    useEffect(() => {
        const loadBusinesses = async () => {
        const daBusinesses: Business[] = await readUnseenBusinesses();
        setBusinesses(daBusinesses);
        }
        loadBusinesses();
    }, []);

    return (
        <div className="bg-[#f4f4f4]">
            <div className="flex justify-center items-center min-h-screen">
                <BusinessExplore businesses={businesses} />
            </div>
        </div>
    );
}